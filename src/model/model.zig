const std = @import("std");
const tensor_mod = @import("../tensor/tensor.zig");
const dtype_mod = @import("../tensor/dtype.zig");
const config_mod = @import("../util/config.zig");
const nn_mod = @import("../nn/layers.zig");
const efla_mod = @import("efla.zig");
const prism_mod = @import("prism.zig");

pub const Tensor = tensor_mod.Tensor;
pub const Shape = tensor_mod.Shape;
pub const DType = dtype_mod.DType;
pub const BF16 = dtype_mod.BF16;

pub const ModelError = error{
    InvalidConfiguration,
    ShapeMismatch,
    DTypeMismatch,
    DeviceMismatch,
    UnsupportedDevice,
    InvalidInputRank,
    UnsupportedOperation,
};

pub const TransformerBlockForwardResult = struct {
    output: *Tensor,
    new_efla_state: ?*efla_mod.EflaState,
    new_prism_state: ?*prism_mod.PrismState,
};

pub const ModelForwardResult = struct {
    logits: *Tensor,
    efla_states: []?*efla_mod.EflaState,
    prism_states: []?*prism_mod.PrismState,

    pub fn deinit(self: *ModelForwardResult, allocator: std.mem.Allocator) void {
        allocator.free(self.efla_states);
        allocator.free(self.prism_states);
        self.logits.deinit();
    }
};

pub const TransformerBlock = struct {
    ln1: *nn_mod.RMSNorm,
    efla: *efla_mod.EflaLayer,
    prism: *prism_mod.PrismLayer,
    ln2: *nn_mod.RMSNorm,
    mlp_up: *nn_mod.Linear,
    mlp_down: *nn_mod.Linear,
    activation: nn_mod.GELU,
    config: config_mod.ModelConfig,
    allocator: std.mem.Allocator,
    device: tensor_mod.Device,
    device_id: i32,

    const Self = @This();

    pub fn init(
        allocator: std.mem.Allocator,
        config: config_mod.ModelConfig,
        device: tensor_mod.Device,
        device_id: i32,
        rng: *std.Random,
    ) !*Self {
        if (config.hidden_dim == 0 or config.intermediate_dim == 0 or config.num_heads == 0 or config.head_dim == 0) {
            return ModelError.InvalidConfiguration;
        }
        if (config.hidden_dim % config.num_heads != 0) {
            return ModelError.InvalidConfiguration;
        }

        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        const ln1 = try nn_mod.RMSNorm.init(allocator, config.hidden_dim, device, device_id);
        errdefer ln1.deinit();

        const efla = try efla_mod.EflaLayer.init(
            allocator,
            config.efla,
            config.hidden_dim,
            config.num_heads,
            config.head_dim,
            device,
            device_id,
            rng,
        );
        errdefer efla.deinit();

        const prism = try prism_mod.PrismLayer.init(
            allocator,
            config.prism,
            config.hidden_dim,
            config.head_dim,
            device,
            device_id,
            rng,
        );
        errdefer prism.deinit();

        const ln2 = try nn_mod.RMSNorm.init(allocator, config.hidden_dim, device, device_id);
        errdefer ln2.deinit();

        const mlp_up = try nn_mod.Linear.init(
            allocator,
            config.hidden_dim,
            config.intermediate_dim,
            false,
            device,
            device_id,
            rng,
        );
        errdefer mlp_up.deinit();

        const mlp_down = try nn_mod.Linear.init(
            allocator,
            config.intermediate_dim,
            config.hidden_dim,
            false,
            device,
            device_id,
            rng,
        );
        errdefer mlp_down.deinit();

        self.* = .{
            .ln1 = ln1,
            .efla = efla,
            .prism = prism,
            .ln2 = ln2,
            .mlp_up = mlp_up,
            .mlp_down = mlp_down,
            .activation = nn_mod.GELU.init(true),
            .config = config,
            .allocator = allocator,
            .device = device,
            .device_id = device_id,
        };

        return self;
    }

    pub fn deinit(self: *Self) void {
        self.ln1.deinit();
        self.efla.deinit();
        self.prism.deinit();
        self.ln2.deinit();
        self.mlp_up.deinit();
        self.mlp_down.deinit();
        self.allocator.destroy(self);
    }

    pub fn forward(
        self: *Self,
        input: *Tensor,
        efla_state: ?*efla_mod.EflaState,
        prism_state: ?*prism_mod.PrismState,
    ) !TransformerBlockForwardResult {
        try self.validateTensorForBlock(input);

        const normed = try self.ln1.forward(input);
        defer normed.deinit();

        var efla_result = try self.efla.forward(normed, efla_state);
        errdefer {
            efla_result.output.deinit();
            if (efla_result.new_state) |state| {
                state.deinit();
            }
        }
        defer efla_result.output.deinit();

        var prism_result = try self.prism.forward(normed, efla_result.output, prism_state);
        errdefer {
            prism_result.output.deinit();
            if (prism_result.new_state) |state| {
                state.deinit();
            }
        }

        const residual = try self.addTensors(input, prism_result.output);
        errdefer residual.deinit();
        prism_result.output.deinit();

        const normed2 = try self.ln2.forward(residual);
        defer normed2.deinit();

        const up = try self.mlp_up.forward(normed2);
        defer up.deinit();

        const activated = try self.activation.forward(self.allocator, up);
        defer activated.deinit();

        const down = try self.mlp_down.forward(activated);
        defer down.deinit();

        const output = try self.addTensors(residual, down);
        residual.deinit();

        return .{
            .output = output,
            .new_efla_state = efla_result.new_state,
            .new_prism_state = prism_result.new_state,
        };
    }

    fn validateTensorForBlock(self: *Self, tensor: *Tensor) !void {
        if (tensor.dtype != .bf16) {
            return ModelError.DTypeMismatch;
        }
        if (tensor.device != self.device or tensor.device_id != self.device_id) {
            return ModelError.DeviceMismatch;
        }
        if (self.device != .cpu) {
            return ModelError.UnsupportedDevice;
        }
    }

    fn addTensors(self: *Self, a: *Tensor, b: *Tensor) !*Tensor {
        try self.validateTensorForBlock(a);
        try self.validateTensorForBlock(b);

        if (!a.shape.equalTo(b.shape)) {
            return ModelError.ShapeMismatch;
        }

        const a_ptr_opt = a.typedPtr(BF16);
        const b_ptr_opt = b.typedPtr(BF16);
        if (a_ptr_opt == null or b_ptr_opt == null) {
            return ModelError.UnsupportedOperation;
        }

        const output = try Tensor.init(self.allocator, a.shape, a.dtype, self.device, self.device_id);
        errdefer output.deinit();

        const o_ptr_opt = output.typedPtr(BF16);
        if (o_ptr_opt == null) {
            return ModelError.UnsupportedOperation;
        }

        const a_ptr = a_ptr_opt.?;
        const b_ptr = b_ptr_opt.?;
        const o_ptr = o_ptr_opt.?;
        const numel = a.shape.numel();

        for (0..numel) |i| {
            o_ptr[i] = BF16.fromFloat32(a_ptr[i].toFloat32() + b_ptr[i].toFloat32());
        }

        return output;
    }

    pub fn backward(self: *Self, grad_output: *Tensor) !*Tensor {
        _ = self;
        return grad_output.clone();
    }
};

pub const EflaModel = struct {
    embed_tokens: *nn_mod.Embedding,
    blocks: []*TransformerBlock,
    final_norm: *nn_mod.RMSNorm,
    lm_head: *nn_mod.Linear,
    config: config_mod.ModelConfig,
    allocator: std.mem.Allocator,
    device: tensor_mod.Device,
    device_id: i32,
    tied_embeddings: bool,

    const Self = @This();

    pub fn init(
        allocator: std.mem.Allocator,
        config: config_mod.ModelConfig,
        device: tensor_mod.Device,
        device_id: i32,
        rng: *std.Random,
    ) !*Self {
        if (config.hidden_dim == 0 or config.intermediate_dim == 0 or config.num_layers == 0 or config.vocab_size == 0) {
            return ModelError.InvalidConfiguration;
        }
        if (config.num_heads == 0 or config.head_dim == 0) {
            return ModelError.InvalidConfiguration;
        }
        if (config.hidden_dim % config.num_heads != 0) {
            return ModelError.InvalidConfiguration;
        }

        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        const embed_tokens = try nn_mod.Embedding.init(
            allocator,
            config.vocab_size,
            config.hidden_dim,
            device,
            device_id,
            rng,
        );
        errdefer embed_tokens.deinit();

        const blocks = try allocator.alloc(*TransformerBlock, config.num_layers);
        errdefer allocator.free(blocks);

        var initialized_blocks: usize = 0;
        errdefer {
            for (blocks[0..initialized_blocks]) |block| {
                block.deinit();
            }
        }

        for (0..config.num_layers) |i| {
            blocks[i] = try TransformerBlock.init(
                allocator,
                config,
                device,
                device_id,
                rng,
            );
            initialized_blocks += 1;
        }

        const final_norm = try nn_mod.RMSNorm.init(allocator, config.hidden_dim, device, device_id);
        errdefer final_norm.deinit();

        const lm_head = try nn_mod.Linear.init(
            allocator,
            config.hidden_dim,
            config.vocab_size,
            false,
            device,
            device_id,
            rng,
        );
        errdefer lm_head.deinit();

        self.* = .{
            .embed_tokens = embed_tokens,
            .blocks = blocks,
            .final_norm = final_norm,
            .lm_head = lm_head,
            .config = config,
            .allocator = allocator,
            .device = device,
            .device_id = device_id,
            .tied_embeddings = config.tie_embeddings,
        };

        return self;
    }

    pub fn deinit(self: *Self) void {
        self.embed_tokens.deinit();
        for (self.blocks) |block| {
            block.deinit();
        }
        self.allocator.free(self.blocks);
        self.final_norm.deinit();
        self.lm_head.deinit();
        self.allocator.destroy(self);
    }

    pub fn forward(self: *Self, input_ids: *Tensor) !*Tensor {
        var result = try self.forwardWithStates(input_ids, null, null);
        defer self.allocator.free(result.efla_states);
        defer self.allocator.free(result.prism_states);
        return result.logits;
    }

    pub fn forwardWithStates(
        self: *Self,
        input_ids: *Tensor,
        efla_states_in: ?[]const ?*efla_mod.EflaState,
        prism_states_in: ?[]const ?*prism_mod.PrismState,
    ) !ModelForwardResult {
        if (input_ids.device != self.device or input_ids.device_id != self.device_id) {
            return ModelError.DeviceMismatch;
        }
        if (self.device != .cpu) {
            return ModelError.UnsupportedDevice;
        }

        const embeds = try self.embed_tokens.forward(input_ids);
        errdefer embeds.deinit();

        var hidden = embeds;
        errdefer hidden.deinit();

        const efla_states_out = try self.allocator.alloc(?*efla_mod.EflaState, self.blocks.len);
        errdefer self.allocator.free(efla_states_out);

        const prism_states_out = try self.allocator.alloc(?*prism_mod.PrismState, self.blocks.len);
        errdefer self.allocator.free(prism_states_out);

        for (0..self.blocks.len) |i| {
            efla_states_out[i] = null;
            prism_states_out[i] = null;
        }

        errdefer {
            for (efla_states_out) |state| {
                if (state) |s| {
                    s.deinit();
                }
            }
            for (prism_states_out) |state| {
                if (state) |s| {
                    s.deinit();
                }
            }
        }

        for (self.blocks, 0..) |block, i| {
            const efla_state = if (efla_states_in) |states|
                if (i < states.len) states[i] else null
            else
                null;

            const prism_state = if (prism_states_in) |states|
                if (i < states.len) states[i] else null
            else
                null;

            const result = try block.forward(hidden, efla_state, prism_state);
            hidden.deinit();
            hidden = result.output;
            efla_states_out[i] = result.new_efla_state;
            prism_states_out[i] = result.new_prism_state;
        }

        const normed = try self.final_norm.forward(hidden);
        hidden.deinit();

        const logits = try self.lm_head.forward(normed);
        normed.deinit();

        return .{
            .logits = logits,
            .efla_states = efla_states_out,
            .prism_states = prism_states_out,
        };
    }

    pub fn backward(self: *Self, grad_output: *Tensor) !*Tensor {
        _ = self;
        return grad_output.clone();
    }

    pub fn collectParameters(self: *Self, allocator: std.mem.Allocator) ![]*Tensor {
        var params = std.ArrayList(*Tensor).init(allocator);
        errdefer params.deinit();

        try params.append(self.embed_tokens.weight);

        for (self.blocks) |block| {
            try params.append(block.ln1.weight);
            try params.append(block.efla.w_k);
            try params.append(block.efla.w_v);
            try params.append(block.efla.w_o);
            if (block.efla.beta_param) |beta_param| {
                try params.append(beta_param);
            }
            for (block.prism.w_beta) |w| {
                try params.append(w);
            }
            for (block.prism.w_k) |w| {
                try params.append(w);
            }
            for (block.prism.w_p) |w| {
                try params.append(w);
            }
            try params.append(block.prism.shortconv.weight);
            try params.append(block.ln2.weight);
            try params.append(block.mlp_up.weight);
            try params.append(block.mlp_down.weight);
        }

        try params.append(self.final_norm.weight);
        if (!self.tied_embeddings) {
            try params.append(self.lm_head.weight);
        }

        return params.toOwnedSlice();
    }

    pub fn getParameterNames(self: *Self, allocator: std.mem.Allocator) ![]const []const u8 {
        var names = std.ArrayList([]const u8).init(allocator);
        errdefer {
            for (names.items) |name| {
                allocator.free(name);
            }
            names.deinit();
        }

        try names.append(try allocator.dupe(u8, "embed_tokens.weight"));

        for (self.blocks, 0..) |block, i| {
            try names.append(try std.fmt.allocPrint(allocator, "blocks.{d}.ln1.weight", .{i}));
            try names.append(try std.fmt.allocPrint(allocator, "blocks.{d}.efla.w_k", .{i}));
            try names.append(try std.fmt.allocPrint(allocator, "blocks.{d}.efla.w_v", .{i}));
            try names.append(try std.fmt.allocPrint(allocator, "blocks.{d}.efla.w_o", .{i}));
            if (block.efla.beta_param != null) {
                try names.append(try std.fmt.allocPrint(allocator, "blocks.{d}.efla.beta_param", .{i}));
            }
            for (0..block.prism.w_beta.len) |j| {
                try names.append(try std.fmt.allocPrint(allocator, "blocks.{d}.prism.w_beta.{d}", .{ i, j }));
            }
            for (0..block.prism.w_k.len) |j| {
                try names.append(try std.fmt.allocPrint(allocator, "blocks.{d}.prism.w_k.{d}", .{ i, j }));
            }
            for (0..block.prism.w_p.len) |j| {
                try names.append(try std.fmt.allocPrint(allocator, "blocks.{d}.prism.w_p.{d}", .{ i, j }));
            }
            try names.append(try std.fmt.allocPrint(allocator, "blocks.{d}.prism.shortconv.weight", .{i}));
            try names.append(try std.fmt.allocPrint(allocator, "blocks.{d}.ln2.weight", .{i}));
            try names.append(try std.fmt.allocPrint(allocator, "blocks.{d}.mlp_up.weight", .{i}));
            try names.append(try std.fmt.allocPrint(allocator, "blocks.{d}.mlp_down.weight", .{i}));
        }

        try names.append(try allocator.dupe(u8, "final_norm.weight"));
        if (!self.tied_embeddings) {
            try names.append(try allocator.dupe(u8, "lm_head.weight"));
        }

        return names.toOwnedSlice();
    }

    pub fn collectGradients(self: *Self, allocator: std.mem.Allocator) ![]*Tensor {
        return self.collectParameters(allocator);
    }

    pub fn countParameters(self: *Self) u64 {
        var count: u64 = 0;
        count += tensorNumel(self.embed_tokens.weight);

        for (self.blocks) |block| {
            count += tensorNumel(block.ln1.weight);
            count += tensorNumel(block.efla.w_k);
            count += tensorNumel(block.efla.w_v);
            count += tensorNumel(block.efla.w_o);
            if (block.efla.beta_param) |beta_param| {
                count += tensorNumel(beta_param);
            }
            for (block.prism.w_beta) |w| {
                count += tensorNumel(w);
            }
            for (block.prism.w_k) |w| {
                count += tensorNumel(w);
            }
            for (block.prism.w_p) |w| {
                count += tensorNumel(w);
            }
            count += tensorNumel(block.prism.shortconv.weight);
            count += tensorNumel(block.ln2.weight);
            count += tensorNumel(block.mlp_up.weight);
            count += tensorNumel(block.mlp_down.weight);
        }

        count += tensorNumel(self.final_norm.weight);
        if (!self.tied_embeddings) {
            count += tensorNumel(self.lm_head.weight);
        }

        return count;
    }
};

fn tensorNumel(tensor: *Tensor) u64 {
    return @as(u64, tensor.shape.numel());
}

test "EflaModel parameter count matches collected parameters and names" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const status = gpa.deinit();
        std.testing.expect(status == .ok) catch unreachable;
    }

    var prng = std.Random.DefaultPrng.init(42);
    var rng = prng.random();

    var config = config_mod.ModelConfig.default1T(gpa.allocator());
    config.hidden_dim = 256;
    config.num_layers = 2;
    config.intermediate_dim = 512;
    config.num_heads = 4;
    config.head_dim = config.hidden_dim / config.num_heads;
    config.tie_embeddings = false;

    const model = try EflaModel.init(gpa.allocator(), config, .cpu, 0, &rng);
    defer model.deinit();

    const params = try model.collectParameters(gpa.allocator());
    defer gpa.allocator().free(params);

    const names = try model.getParameterNames(gpa.allocator());
    defer {
        for (names) |name| {
            gpa.allocator().free(name);
        }
        gpa.allocator().free(names);
    }

    try std.testing.expectEqual(params.len, names.len);

    var manual_count: u64 = 0;
    for (params) |param| {
        manual_count += @as(u64, param.shape.numel());
    }

    try std.testing.expectEqual(manual_count, model.countParameters());
}
