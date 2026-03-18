const std = @import("std");

pub const DType = enum(u8) {
    fp32 = 0,
    fp16 = 1,
    bf16 = 2,
    fp8_e4m3 = 3,
    fp8_e5m2 = 4,
    fp4 = 5,
    int8 = 6,
    int32 = 7,
    int64 = 8,
    bool_ = 9,

    pub fn bitSize(self: DType) usize {
        return switch (self) {
            .fp32 => 32,
            .fp16 => 16,
            .bf16 => 16,
            .fp8_e4m3 => 8,
            .fp8_e5m2 => 8,
            .fp4 => 4,
            .int8 => 8,
            .int32 => 32,
            .int64 => 64,
            .bool_ => 8,
        };
    }

    pub fn isFloatingPoint(self: DType) bool {
        return switch (self) {
            .fp32, .fp16, .bf16, .fp8_e4m3, .fp8_e5m2, .fp4 => true,
            else => false,
        };
    }

    pub fn isQuantized(self: DType) bool {
        return switch (self) {
            .fp8_e4m3, .fp8_e5m2, .fp4, .int8 => true,
            else => false,
        };
    }

    pub fn accumulationDType(self: DType) DType {
        return switch (self) {
            .fp32 => .fp32,
            .fp16, .bf16 => .fp32,
            .fp8_e4m3, .fp8_e5m2 => .fp32,
            .fp4 => .fp32,
            else => .fp32,
        };
    }

    pub fn min(self: DType) f64 {
        return switch (self) {
            .fp32 => -@as(f64, std.math.floatMax(f32)),
            .fp16 => -65504.0,
            .bf16 => -3.3895313892515355e+38,
            .fp8_e4m3 => -448.0,
            .fp8_e5m2 => -57344.0,
            .fp4 => -6.0,
            .int8 => -128.0,
            .int32 => -2147483648.0,
            .int64 => -9223372036854775808.0,
            .bool_ => 0.0,
        };
    }

    pub fn max(self: DType) f64 {
        return switch (self) {
            .fp32 => @as(f64, std.math.floatMax(f32)),
            .fp16 => 65504.0,
            .bf16 => 3.3895313892515355e+38,
            .fp8_e4m3 => 448.0,
            .fp8_e5m2 => 57344.0,
            .fp4 => 6.0,
            .int8 => 127.0,
            .int32 => 2147483647.0,
            .int64 => 9223372036854775807.0,
            .bool_ => 1.0,
        };
    }
};

pub const ScaleFactor = struct {
    scale: f32,
    inv_scale: f32,
    amax: f32,
    amax_history: [16]f32,

    pub fn init() ScaleFactor {
        return .{
            .scale = 1.0,
            .inv_scale = 1.0,
            .amax = 0.0,
            .amax_history = [_]f32{0.0} ** 16,
        };
    }

    pub fn fromScale(scale: f32) ScaleFactor {
        if (std.math.isNan(scale) or std.math.isInf(scale)) {
            return .{
                .scale = 1.0,
                .inv_scale = 1.0,
                .amax = 0.0,
                .amax_history = [_]f32{0.0} ** 16,
            };
        }
        return .{
            .scale = scale,
            .inv_scale = if (scale != 0.0) 1.0 / scale else 0.0,
            .amax = 0.0,
            .amax_history = [_]f32{0.0} ** 16,
        };
    }

    pub fn update(self: *ScaleFactor, new_amax: f32, target_max: f32) void {
        if (std.math.isNan(target_max) or std.math.isInf(target_max) or target_max <= 0.0) {
            return;
        }

        var abs_amax = @abs(new_amax);
        if (std.math.isNan(abs_amax) or std.math.isInf(abs_amax)) {
            abs_amax = 0.0;
        }

        var i: usize = 15;
        while (i > 0) : (i -= 1) {
            self.amax_history[i] = self.amax_history[i - 1];
        }
        self.amax_history[0] = abs_amax;

        var max_amax: f32 = 0.0;
        for (self.amax_history) |a| {
            if (!std.math.isNan(a) and a > max_amax) max_amax = a;
        }

        self.amax = max_amax;

        if (max_amax > 0.0) {
            self.scale = target_max / max_amax;
            self.inv_scale = max_amax / target_max;
        }
    }
};

pub const BlockScale = struct {
    block_size: u32,
    num_blocks: u32,
    scales: []ScaleFactor,

    pub fn init(allocator: std.mem.Allocator, total_elements: usize, block_size: usize) !BlockScale {
        if (block_size == 0) return error.ZeroBlockSize;
        if (total_elements == 0) return error.ZeroElements;
        if (@bitSizeOf(usize) > 32 and block_size > std.math.maxInt(u32)) return error.BlockSizeTooLarge;

        const num_blocks = blk: {
            const sum = @addWithOverflow(total_elements, block_size - 1);
            if (sum[1] != 0) return error.Overflow;
            break :blk sum[0] / block_size;
        };
        if (num_blocks > std.math.maxInt(u32)) return error.NumBlocksTooLarge;

        const scales = try allocator.alloc(ScaleFactor, num_blocks);

        for (scales) |*s| {
            s.* = ScaleFactor.init();
        }

        return .{
            .block_size = @intCast(block_size),
            .num_blocks = @intCast(num_blocks),
            .scales = scales,
        };
    }

    pub fn deinit(self: *BlockScale, allocator: std.mem.Allocator) void {
        allocator.free(self.scales);
        self.scales = &[_]ScaleFactor{};
        self.num_blocks = 0;
    }

    pub fn getScale(self: BlockScale, block_idx: usize) !*ScaleFactor {
        if (block_idx >= self.scales.len) return error.OutOfBounds;
        return &self.scales[block_idx];
    }
};

pub const FP8_E4M3 = packed struct(u8) {
    mantissa: u3,
    exponent: u4,
    sign: u1,

    pub const MAX_VALUE: f32 = 448.0;
    pub const MIN_POSITIVE: f32 = 0.001953125;

    pub fn fromFloat32(val: f32) FP8_E4M3 {
        const bits: u32 = @bitCast(val);
        const sign: u1 = @intCast(bits >> 31);

        if (std.math.isNan(val)) return .{ .sign = sign, .exponent = 15, .mantissa = 7 };
        if (std.math.isInf(val)) return .{ .sign = sign, .exponent = 15, .mantissa = 6 };

        const exp_bits: u32 = (bits >> 23) & 0xFF;
        const mant_bits: u32 = bits & 0x7FFFFF;

        if (exp_bits == 0 and mant_bits == 0) {
            return .{ .sign = sign, .exponent = 0, .mantissa = 0 };
        }

        var actual_exp: i32 = undefined;
        var full_mant: u32 = undefined;
        if (exp_bits == 0) {
            actual_exp = -126;
            full_mant = mant_bits;
        } else {
            actual_exp = @as(i32, @intCast(exp_bits)) - 127;
            full_mant = 0x800000 | mant_bits;
        }

        var fp8_exp: i32 = actual_exp + 7;
        var fp8_mant: u32 = undefined;
        var round_bit: u32 = 0;
        var sticky_bit: u32 = 0;

        if (fp8_exp <= 0) {
            const raw_shift = 21 - fp8_exp;
            const shift: u5 = @intCast(if (raw_shift > 31) @as(i32, 31) else if (raw_shift < 0) @as(i32, 0) else raw_shift);

            fp8_mant = if (shift >= 24) 0 else full_mant >> shift;
            if (shift > 0) {
                const rs: u5 = @intCast(if (shift - 1 > 31) @as(i32, 31) else shift - 1);
                round_bit = if (rs >= 24) 0 else (full_mant >> rs) & 1;
                const mask: u32 = if (rs == 0) 0 else (@as(u32, 1) << rs) -% 1;
                sticky_bit = if ((full_mant & mask) != 0) 1 else 0;
            }
            fp8_exp = 0;
        } else {
            round_bit = (mant_bits >> 19) & 1;
            const mask: u32 = (@as(u32, 1) << 19) - 1;
            sticky_bit = if ((mant_bits & mask) != 0) 1 else 0;
            fp8_mant = mant_bits >> 20;
        }

        if (round_bit == 1 and (sticky_bit == 1 or (fp8_mant & 1) == 1)) {
            fp8_mant += 1;
            if (fp8_exp == 0) {
                if (fp8_mant == 8) {
                    fp8_exp = 1;
                    fp8_mant = 0;
                }
            } else {
                if (fp8_mant == 8) {
                    fp8_exp += 1;
                    fp8_mant = 0;
                }
            }
        }

        if (fp8_exp > 15 or (fp8_exp == 15 and fp8_mant >= 7)) {
            fp8_exp = 15;
            fp8_mant = 6;
        }

        return .{ .sign = sign, .exponent = @intCast(fp8_exp), .mantissa = @intCast(fp8_mant) };
    }

    pub fn toFloat32(self: FP8_E4M3) f32 {
        if (self.exponent == 15 and self.mantissa == 7) {
            const bits: u32 = (@as(u32, self.sign) << 31) | 0x7FC00000;
            return @bitCast(bits);
        }
        if (self.exponent == 0) {
            if (self.mantissa == 0) {
                const bits: u32 = @as(u32, self.sign) << 31;
                return @bitCast(bits);
            }
            var m: u32 = self.mantissa;
            var e: i32 = -6;
            while ((m & 8) == 0) {
                m <<= 1;
                e -= 1;
            }
            m &= 7;
            const fp32_exp: u32 = @intCast(e + 127);
            const fp32_mant: u32 = m << 20;
            const bits: u32 = (@as(u32, self.sign) << 31) | (fp32_exp << 23) | fp32_mant;
            return @bitCast(bits);
        }

        const fp32_exp: u32 = @intCast(@as(i32, self.exponent) - 7 + 127);
        const fp32_mant: u32 = @as(u32, self.mantissa) << 20;
        const bits: u32 = (@as(u32, self.sign) << 31) | (fp32_exp << 23) | fp32_mant;
        return @bitCast(bits);
    }
};

pub const FP8_E5M2 = packed struct(u8) {
    mantissa: u2,
    exponent: u5,
    sign: u1,

    pub const MAX_VALUE: f32 = 57344.0;
    pub const MIN_POSITIVE: f32 = 0.0000152587890625;

    pub fn fromFloat32(val: f32) FP8_E5M2 {
        const bits: u32 = @bitCast(val);
        const sign: u1 = @intCast(bits >> 31);

        if (std.math.isNan(val)) return .{ .sign = sign, .exponent = 31, .mantissa = 1 };
        if (std.math.isInf(val)) return .{ .sign = sign, .exponent = 31, .mantissa = 0 };

        const exp_bits: u32 = (bits >> 23) & 0xFF;
        const mant_bits: u32 = bits & 0x7FFFFF;

        if (exp_bits == 0 and mant_bits == 0) {
            return .{ .sign = sign, .exponent = 0, .mantissa = 0 };
        }

        var actual_exp: i32 = undefined;
        var full_mant: u32 = undefined;
        if (exp_bits == 0) {
            actual_exp = -126;
            full_mant = mant_bits;
        } else {
            actual_exp = @as(i32, @intCast(exp_bits)) - 127;
            full_mant = 0x800000 | mant_bits;
        }

        var fp8_exp: i32 = actual_exp + 15;
        var fp8_mant: u32 = undefined;
        var round_bit: u32 = 0;
        var sticky_bit: u32 = 0;

        if (fp8_exp <= 0) {
            const raw_shift = 22 - fp8_exp;
            const shift: u5 = @intCast(if (raw_shift > 31) @as(i32, 31) else if (raw_shift < 0) @as(i32, 0) else raw_shift);

            fp8_mant = if (shift >= 24) 0 else full_mant >> shift;
            if (shift > 0) {
                const rs: u5 = @intCast(if (shift - 1 > 31) @as(i32, 31) else shift - 1);
                round_bit = if (rs >= 24) 0 else (full_mant >> rs) & 1;
                const mask: u32 = if (rs == 0) 0 else (@as(u32, 1) << rs) -% 1;
                sticky_bit = if ((full_mant & mask) != 0) 1 else 0;
            }
            fp8_exp = 0;
        } else {
            round_bit = (mant_bits >> 20) & 1;
            const mask: u32 = (@as(u32, 1) << 20) - 1;
            sticky_bit = if ((mant_bits & mask) != 0) 1 else 0;
            fp8_mant = mant_bits >> 21;
        }

        if (round_bit == 1 and (sticky_bit == 1 or (fp8_mant & 1) == 1)) {
            fp8_mant += 1;
            if (fp8_exp == 0) {
                if (fp8_mant == 4) {
                    fp8_exp = 1;
                    fp8_mant = 0;
                }
            } else {
                if (fp8_mant == 4) {
                    fp8_exp += 1;
                    fp8_mant = 0;
                }
            }
        }

        if (fp8_exp >= 31) {
            return .{ .sign = sign, .exponent = 31, .mantissa = 0 };
        }

        return .{ .sign = sign, .exponent = @intCast(fp8_exp), .mantissa = @intCast(fp8_mant) };
    }

    pub fn toFloat32(self: FP8_E5M2) f32 {
        if (self.exponent == 31) {
            if (self.mantissa == 0) {
                const bits: u32 = (@as(u32, self.sign) << 31) | 0x7F800000;
                return @bitCast(bits);
            } else {
                const bits: u32 = (@as(u32, self.sign) << 31) | 0x7FC00000;
                return @bitCast(bits);
            }
        }
        if (self.exponent == 0) {
            if (self.mantissa == 0) {
                const bits: u32 = @as(u32, self.sign) << 31;
                return @bitCast(bits);
            }
            var m: u32 = self.mantissa;
            var e: i32 = -14;
            while ((m & 4) == 0) {
                m <<= 1;
                e -= 1;
            }
            m &= 3;
            const fp32_exp: u32 = @intCast(e + 127);
            const fp32_mant: u32 = m << 21;
            const bits: u32 = (@as(u32, self.sign) << 31) | (fp32_exp << 23) | fp32_mant;
            return @bitCast(bits);
        }

        const fp32_exp: u32 = @intCast(@as(i32, self.exponent) - 15 + 127);
        const fp32_mant: u32 = @as(u32, self.mantissa) << 21;
        const bits: u32 = (@as(u32, self.sign) << 31) | (fp32_exp << 23) | fp32_mant;
        return @bitCast(bits);
    }
};

pub const BF16 = packed struct(u16) {
    mantissa: u7,
    exponent: u8,
    sign: u1,

    pub fn fromFloat32(val: f32) BF16 {
        const bits: u32 = @bitCast(val);
        if (std.math.isNan(val)) {
            const upper = (bits >> 16) | 0x0040 | 1;
            return @bitCast(@as(u16, @truncate(upper)));
        }
        const round_bit = (bits >> 15) & 1;
        const sticky_bit = if ((bits & 0x7FFF) != 0) @as(u32, 1) else 0;
        const lsb = (bits >> 16) & 1;

        var upper: u32 = bits >> 16;
        if (round_bit == 1 and (sticky_bit == 1 or lsb == 1)) {
            upper += 1;
            const new_exp = (upper >> 7) & 0xFF;
            if (new_exp == 0xFF) {
                const was_inf_or_nan = ((bits >> 23) & 0xFF) == 0xFF;
                if (!was_inf_or_nan) {
                    if ((upper & 0x8000) != 0) {
                        return @bitCast(@as(u16, 0xFF80));
                    } else {
                        return @bitCast(@as(u16, 0x7F80));
                    }
                }
            }
        }
        return @bitCast(@as(u16, @truncate(upper)));
    }

    pub fn toFloat32(self: BF16) f32 {
        const bits: u32 = @as(u32, @bitCast(self)) << 16;
        return @bitCast(bits);
    }
};

pub const FP16 = packed struct(u16) {
    mantissa: u10,
    exponent: u5,
    sign: u1,

    pub fn fromFloat32(val: f32) FP16 {
        const bits: u32 = @bitCast(val);
        const sign: u1 = @intCast(bits >> 31);

        if (std.math.isNan(val)) return .{ .sign = sign, .exponent = 31, .mantissa = 1 };
        if (std.math.isInf(val)) return .{ .sign = sign, .exponent = 31, .mantissa = 0 };

        const exp_bits: u32 = (bits >> 23) & 0xFF;
        const mant_bits: u32 = bits & 0x7FFFFF;

        if (exp_bits == 0 and mant_bits == 0) {
            return .{ .sign = sign, .exponent = 0, .mantissa = 0 };
        }

        var actual_exp: i32 = undefined;
        var full_mant: u32 = undefined;
        if (exp_bits == 0) {
            actual_exp = -126;
            full_mant = mant_bits;
        } else {
            actual_exp = @as(i32, @intCast(exp_bits)) - 127;
            full_mant = 0x800000 | mant_bits;
        }

        var fp16_exp: i32 = actual_exp + 15;
        var fp16_mant: u32 = undefined;
        var round_bit: u32 = 0;
        var sticky_bit: u32 = 0;

        if (fp16_exp <= 0) {
            const raw_shift = 14 - fp16_exp;
            const shift: u5 = @intCast(if (raw_shift > 31) @as(i32, 31) else if (raw_shift < 0) @as(i32, 0) else raw_shift);

            fp16_mant = if (shift >= 24) 0 else full_mant >> shift;
            if (shift > 0) {
                const rs: u5 = @intCast(if (shift - 1 > 31) @as(i32, 31) else shift - 1);
                round_bit = if (rs >= 24) 0 else (full_mant >> rs) & 1;
                const mask: u32 = if (rs == 0) 0 else (@as(u32, 1) << rs) -% 1;
                sticky_bit = if ((full_mant & mask) != 0) 1 else 0;
            }
            fp16_exp = 0;
        } else {
            round_bit = (mant_bits >> 12) & 1;
            const mask: u32 = (@as(u32, 1) << 12) - 1;
            sticky_bit = if ((mant_bits & mask) != 0) 1 else 0;
            fp16_mant = mant_bits >> 13;
        }

        if (round_bit == 1 and (sticky_bit == 1 or (fp16_mant & 1) == 1)) {
            fp16_mant += 1;
            if (fp16_exp == 0) {
                if (fp16_mant == 1024) {
                    fp16_exp = 1;
                    fp16_mant = 0;
                }
            } else {
                if (fp16_mant == 1024) {
                    fp16_exp += 1;
                    fp16_mant = 0;
                }
            }
        }

        if (fp16_exp >= 31) {
            return .{ .sign = sign, .exponent = 31, .mantissa = 0 };
        }

        return .{ .sign = sign, .exponent = @intCast(fp16_exp), .mantissa = @intCast(fp16_mant) };
    }

    pub fn toFloat32(self: FP16) f32 {
        if (self.exponent == 31) {
            if (self.mantissa == 0) {
                const bits: u32 = (@as(u32, self.sign) << 31) | 0x7F800000;
                return @bitCast(bits);
            } else {
                const bits: u32 = (@as(u32, self.sign) << 31) | 0x7FC00000;
                return @bitCast(bits);
            }
        }
        if (self.exponent == 0) {
            if (self.mantissa == 0) {
                const bits: u32 = @as(u32, self.sign) << 31;
                return @bitCast(bits);
            }
            var m: u32 = self.mantissa;
            var e: i32 = -14;
            while ((m & 1024) == 0) {
                m <<= 1;
                e -= 1;
            }
            m &= 1023;
            const fp32_exp: u32 = @intCast(e + 127);
            const fp32_mant: u32 = m << 13;
            const bits: u32 = (@as(u32, self.sign) << 31) | (fp32_exp << 23) | fp32_mant;
            return @bitCast(bits);
        }

        const fp32_exp: u32 = @intCast(@as(i32, self.exponent) - 15 + 127);
        const fp32_mant: u32 = @as(u32, self.mantissa) << 13;
        const bits: u32 = (@as(u32, self.sign) << 31) | (fp32_exp << 23) | fp32_mant;
        return @bitCast(bits);
    }
};

fn readUnaligned(comptime T: type, bytes: []const u8, index: usize) T {
    const size = @sizeOf(T);
    const product = @mulWithOverflow(index, size);
    if (product[1] != 0) @panic("readUnaligned: index overflow");
    const start = product[0];
    if (start + size > bytes.len) @panic("readUnaligned: out of bounds");
    var val: T = undefined;
    @memcpy(std.mem.asBytes(&val), bytes[start .. start + size]);
    return val;
}

fn writeUnaligned(comptime T: type, bytes: []u8, index: usize, val: T) void {
    const size = @sizeOf(T);
    const product = @mulWithOverflow(index, size);
    if (product[1] != 0) @panic("writeUnaligned: index overflow");
    const start = product[0];
    if (start + size > bytes.len) @panic("writeUnaligned: out of bounds");
    @memcpy(bytes[start .. start + size], std.mem.asBytes(&val));
}

fn readFP4(bytes: []const u8, index: usize) f32 {
    const byte_idx = index / 2;
    const is_high = (index % 2) != 0;
    const b = bytes[byte_idx];
    const nibble = if (is_high) b >> 4 else b & 0x0F;

    const sign: f32 = if ((nibble & 8) != 0) -1.0 else 1.0;
    const exp = (nibble >> 1) & 3;
    const mant = nibble & 1;
    if (exp == 0) {
        return sign * @as(f32, @floatFromInt(mant)) * 0.5;
    }
    const e_val: i32 = @as(i32, exp) - 1;
    const mant_val = 1.0 + @as(f32, @floatFromInt(mant)) * 0.5;
    return sign * mant_val * std.math.ldexp(@as(f32, 1.0), e_val);
}

fn writeFP4(bytes: []u8, index: usize, val: f32) void {
    const byte_idx = index / 2;
    const is_high = (index % 2) != 0;

    var nibble: u8 = 0;
    if (std.math.isNan(val)) {
        nibble = 0;
    } else if (val == 0.0) {
        nibble = 0;
    } else {
        const sign: u8 = if (val < 0.0) 8 else 0;
        const abs_val = @abs(val);
        if (abs_val >= 6.0) {
            nibble = sign | 7;
        } else if (abs_val < 0.25) {
            nibble = sign;
        } else if (abs_val < 0.75) {
            nibble = sign | 1;
        } else if (abs_val < 1.25) {
            nibble = sign | 2;
        } else if (abs_val < 1.75) {
            nibble = sign | 3;
        } else if (abs_val < 2.5) {
            nibble = sign | 4;
        } else if (abs_val < 3.5) {
            nibble = sign | 5;
        } else if (abs_val < 5.0) {
            nibble = sign | 6;
        } else {
            nibble = sign | 7;
        }
    }

    if (is_high) {
        bytes[byte_idx] = (bytes[byte_idx] & 0x0F) | (nibble << 4);
    } else {
        bytes[byte_idx] = (bytes[byte_idx] & 0xF0) | nibble;
    }
}

pub fn castSlice(comptime from: DType, comptime to: DType, input: []const u8, output: []u8) !void {
    const from_bits = from.bitSize();
    const to_bits = to.bitSize();

    if (from_bits >= 8) {
        if ((input.len * 8) % from_bits != 0) return error.InvalidInputLength;
    }
    const n = (input.len * 8) / from_bits;

    if (to_bits >= 8) {
        if ((output.len * 8) % to_bits != 0) return error.InvalidOutputLength;
        if ((output.len * 8) / to_bits != n) return error.InvalidOutputLength;
    } else {
        if ((n + 1) / 2 > output.len) return error.InvalidOutputLength;
    }

    const in_start = @intFromPtr(input.ptr);
    const in_end = in_start +% input.len;
    const out_start = @intFromPtr(output.ptr);
    const out_end = out_start +% output.len;
    if (in_start < out_end and out_start < in_end) {
        if (input.ptr != output.ptr or input.len != output.len or from != to) {
            return error.OverlappingMemory;
        }
    }

    if (from == to and from_bits >= 8) {
        if (input.ptr != output.ptr) {
            @memcpy(output[0..input.len], input);
        }
        return;
    }

    for (0..n) |i| {
        const val_f64: f64 = switch (from) {
            .fp32 => @as(f64, readUnaligned(f32, input, i)),
            .fp16 => @as(f64, readUnaligned(FP16, input, i).toFloat32()),
            .bf16 => @as(f64, readUnaligned(BF16, input, i).toFloat32()),
            .fp8_e4m3 => @as(f64, readUnaligned(FP8_E4M3, input, i).toFloat32()),
            .fp8_e5m2 => @as(f64, readUnaligned(FP8_E5M2, input, i).toFloat32()),
            .fp4 => @as(f64, readFP4(input, i)),
            .int8 => @floatFromInt(readUnaligned(i8, input, i)),
            .int32 => @floatFromInt(readUnaligned(i32, input, i)),
            .int64 => @floatFromInt(readUnaligned(i64, input, i)),
            .bool_ => if (readUnaligned(u8, input, i) != 0) @as(f64, 1.0) else @as(f64, 0.0),
        };

        switch (to) {
            .fp32 => writeUnaligned(f32, output, i, @floatCast(val_f64)),
            .fp16 => writeUnaligned(FP16, output, i, FP16.fromFloat32(@floatCast(val_f64))),
            .bf16 => writeUnaligned(BF16, output, i, BF16.fromFloat32(@floatCast(val_f64))),
            .fp8_e4m3 => writeUnaligned(FP8_E4M3, output, i, FP8_E4M3.fromFloat32(@floatCast(val_f64))),
            .fp8_e5m2 => writeUnaligned(FP8_E5M2, output, i, FP8_E5M2.fromFloat32(@floatCast(val_f64))),
            .fp4 => writeFP4(output, i, @floatCast(val_f64)),
            .int8 => {
                if (std.math.isNan(val_f64)) {
                    writeUnaligned(i8, output, i, 0);
                } else if (val_f64 >= 127.0) {
                    writeUnaligned(i8, output, i, 127);
                } else if (val_f64 <= -128.0) {
                    writeUnaligned(i8, output, i, -128);
                } else {
                    writeUnaligned(i8, output, i, @intFromFloat(val_f64));
                }
            },
            .int32 => {
                if (std.math.isNan(val_f64)) {
                    writeUnaligned(i32, output, i, 0);
                } else if (val_f64 >= 2147483647.0) {
                    writeUnaligned(i32, output, i, std.math.maxInt(i32));
                } else if (val_f64 <= -2147483648.0) {
                    writeUnaligned(i32, output, i, std.math.minInt(i32));
                } else {
                    writeUnaligned(i32, output, i, @intFromFloat(val_f64));
                }
            },
            .int64 => {
                if (std.math.isNan(val_f64)) {
                    writeUnaligned(i64, output, i, 0);
                } else if (val_f64 >= 9223372036854775807.0) {
                    writeUnaligned(i64, output, i, std.math.maxInt(i64));
                } else if (val_f64 <= -9223372036854775808.0) {
                    writeUnaligned(i64, output, i, std.math.minInt(i64));
                } else {
                    writeUnaligned(i64, output, i, @intFromFloat(val_f64));
                }
            },
            .bool_ => {
                if (std.math.isNan(val_f64)) {
                    writeUnaligned(u8, output, i, 0);
                } else {
                    writeUnaligned(u8, output, i, if (val_f64 != 0.0) 1 else 0);
                }
            },
        }
    }
}

test "FP8_E4M3 roundtrip" {
    const test_vals = [_]f32{ 0.0, -0.0, 1.0, -1.0, 448.0, -448.0, 0.5, -0.5, 0.015625, -0.015625, 0.001953125, -0.001953125, std.math.inf(f32), -std.math.inf(f32), std.math.nan(f32) };

    for (test_vals) |v| {
        const fp8 = FP8_E4M3.fromFloat32(v);
        const back = fp8.toFloat32();
        if (std.math.isNan(v)) {
            try std.testing.expect(std.math.isNan(back));
        } else if (std.math.isInf(v)) {
            try std.testing.expectEqual(@as(f32, if (v > 0) 448.0 else -448.0), back);
        } else {
            try std.testing.expectEqual(v, back);
        }
    }
}

test "FP16 roundtrip" {
    const test_vals = [_]f32{ 0.0, -0.0, 1.0, -1.0, 65504.0, -65504.0, 0.5, -0.5, 6.103515625e-5, -6.103515625e-5, 5.960464477539063e-8, -5.960464477539063e-8, std.math.inf(f32), -std.math.inf(f32), std.math.nan(f32) };

    for (test_vals) |v| {
        const fp16 = FP16.fromFloat32(v);
        const back = fp16.toFloat32();
        if (std.math.isNan(v)) {
            try std.testing.expect(std.math.isNan(back));
        } else {
            try std.testing.expectEqual(v, back);
        }
    }
}

test "BF16 roundtrip" {
    const test_vals = [_]f32{ 0.0, -0.0, 1.0, -1.0, 100.0, -100.0, 0.5, -0.5, 1.17549435e-38, -1.17549435e-38, 9.1835496e-41, -9.1835496e-41, std.math.inf(f32), -std.math.inf(f32), std.math.nan(f32) };

    for (test_vals) |v| {
        const bf16 = BF16.fromFloat32(v);
        const back = bf16.toFloat32();
        if (std.math.isNan(v)) {
            try std.testing.expect(std.math.isNan(back));
        } else {
            try std.testing.expectEqual(v, back);
        }
    }
}
