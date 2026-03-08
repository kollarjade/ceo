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

/// Transformer block combining EFLA and PRISM
pub const TransformerBlock = struct {
    /// Pre-attention norm
    ln1: *nn_mod.RMSNorm,
    /// EFLA layer
    efla: *efla_mod.EflaLayer,
    /// PRISM layer
    prism: *prism_mod.PrismLayer,
    /// Pre-FFN norm
    ln2: *nn_mod.RMSNorm,
    /// MLP up projection
    mlp_up: *nn_mod.Linear,
    /// MLP down projection
    mlp_down: *nn_mod.Linear,
    /// Activation function
    activation: nn_mod.GELU,
    /// Configuration
    config: config_mod.ModelConfig,
    /// Allocator
    allocator: std.mem.Allocator,
    /// Device
    device: tensor_mod.Device,
    /// Device ID
    device_id: i32,

    const Self = @This();

    pub fn init(
        allocator: std.mem.Allocator,
        config: config_mod.ModelConfig,
        device: tensor_mod.Device,
        device_id: i32,
        rng: *std.Random,
    ) !*Self {
        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        const hidden_dim = config.hidden_dim;
        const intermediate_dim = config.intermediate_dim;

        var ln1 = try nn_mod.RMSNorm.init(allocator, hidden_dim, device, device_id);
        errdefer ln1.deinit();

        var efla = try efla_mod.EflaLayer.init(
            allocator,
            config.efla,
            hidden_dim,
            config.num_heads,
            config.head_dim,
            device,
            device_id,
            rng,
        );
        errdefer efla.deinit();

        var prism = try prism_mod.PrismLayer.init(
            allocator,
            config.prism,
            hidden_dim,
            config.head_dim,
            device,
            device_id,
            rng,
        );
        errdefer prism.deinit();

        var ln2 = try nn_mod.RMSNorm.init(allocator, hidden_dim, device, device_id);
        errdefer ln2.deinit();

        var mlp_up = try nn_mod.Linear.init(
            allocator,
            hidden_dim,
            intermediate_dim,
            false,
            device,
            device_id,
            rng,
        );
        errdefer mlp_up.deinit();

        var mlp_down = try nn_mod.Linear.init(
            allocator,
            intermediate_dim,
            hidden_dim,
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

    /// Forward pass through block
    pub fn forward(
        self: *Self,
        input: *Tensor,
        efla_state: ?*efla_mod.EflaState,
        prism_state: ?*prism_mod.PrismState,
    ) !struct { output: *Tensor, new_efla_state: *efla_mod.EflaState, new_prism_state: *prism_mod.PrismState } {
        // Pre-norm for attention
        var normed = try self.ln1.forward(input);
        defer normed.deinit();

        // EFLA forward
        var efla_result = try self.efla.forward(normed, efla_state);
        defer efla_result.output.deinit();

        // PRISM forward
        var prism_result = try self.prism.forward(normed, efla_result.output, prism_state);
        defer {
            // Don't deinit prism_result.output if we're using it
        }

        // Residual connection
        var residual = try self.addTensors(input, prism_result.output);
        errdefer residual.deinit();

        // Pre-norm for MLP
        var normed2 = try self.ln2.forward(residual);
        defer normed2.deinit();

        // MLP
        var up = try self.mlp_up.forward(normed2);
        defer up.deinit();

        var activated = try self.activation.forward(self.allocator, up);
        defer activated.deinit();

        var down = try self.mlp_down.forward(activated);
        defer down.deinit();

        // Final residual
        var output = try self.addTensors(residual, down);
        residual.deinit();

        return .{
            .output = output,
            .new_efla_state = efla_result.new_state,
            .new_prism_state = prism_result.new_state,
        };
    }

    fn addTensors(self: *Self, a: *Tensor, b: *Tensor) !*Tensor {
        _ = self;
        std.debug.assert(a.shape.equalTo(b.shape));

        var output = try Tensor.init(self.allocator, a.shape, .bf16, self.device, self.device_id);
        errdefer output.deinit();

        const a_ptr = a.typedPtr(BF16).?;
        const b_ptr = b.typedPtr(BF16).?;
        const o_ptr = output.typedPtr(BF16).?;

        const numel = a.shape.numel();
        for (0..numel) |i| {
            const sum = a_ptr[i].toFloat32() + b_ptr[i].toFloat32();
            o_ptr[i] = BF16.fromFloat32(sum);
        }

        return output;
    }

    /// Backward pass
    pub fn backward(self: *Self, grad_output: *Tensor) !*Tensor {
        _ = self;
        _ = grad_output;

        // Placeholder - full implementation would backprop through all layers
        var grad = try Tensor.zeros(self.allocator, self.config.hidden_dim, .bf16, self.device, self.device_id);
        return grad.reshape(Shape.init(&[_]usize{1}));
    }
};

/// Complete EFLA-PRISM Language Model
pub const EflaModel = struct {
    /// Token embedding
    embed_tokens: *nn_mod.Embedding,
    /// Transformer blocks
    blocks: []*TransformerBlock,
    /// Final normalization
    final_norm: *nn_mod.RMSNorm,
    /// Output projection (LM head)
    lm_head: *nn_mod.Linear,
    /// Configuration
    config: config_mod.ModelConfig,
    /// Allocator
    allocator: std.mem.Allocator,
    /// Device
    device: tensor_mod.Device,
    /// Device ID
    device_id: i32,

    const Self = @This();

    pub fn init(
        allocator: std.mem.Allocator,
        config: config_mod.ModelConfig,
        device: tensor_mod.Device,
        device_id: i32,
        rng: *std.Random,
    ) !*Self {
        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        // Token embedding
        var embed_tokens = try nn_mod.Embedding.init(
            allocator,
            config.vocab_size,
            config.hidden_dim,
            device,
            device_id,
            rng,
        );
        errdefer embed_tokens.deinit();

        // Transformer blocks
        var blocks = try allocator.alloc(*TransformerBlock, config.num_layers);
        errdefer allocator.free(blocks);

        for (0..config.num_layers) |i| {
            blocks[i] = try TransformerBlock.init(
                allocator,
                config,
                device,
                device_id,
                rng,
            );
            std.log.debug("Initialized block {d}/{d}", .{ i + 1, config.num_layers });
        }

        // Final norm
        var final_norm = try nn_mod.RMSNorm.init(allocator, config.hidden_dim, device, device_id);
        errdefer final_norm.deinit();

        // LM head
        var lm_head = if (config.tie_embeddings)
            // Tied embeddings - would share weight with embed_tokens
            try nn_mod.Linear.init(
                allocator,
                config.hidden_dim,
                config.vocab_size,
                false,
                device,
                device_id,
                rng,
            )
        else
            try nn_mod.Linear.init(
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

    /// Forward pass
    /// input_ids: (batch, seq_len) - token indices
    /// Returns: (batch, seq_len, vocab_size) - logits
    pub fn forward(self: *Self, input_ids: *Tensor) !*Tensor {
        // Token embedding
        var embeds = try self.embed_tokens.forward(input_ids);
        defer {
            // Don't deinit if using later
        }

        var hidden = embeds;

        // Pass through transformer blocks
        for (self.blocks) |block| {
            var result = try block.forward(hidden, null, null);
            hidden.deinit();
            hidden = result.output;
            // States would be managed properly in full implementation
        }

        // Final norm
        var normed = try self.final_norm.forward(hidden);
        hidden.deinit();

        // LM head
        var logits = try self.lm_head.forward(normed);
        normed.deinit();

        return logits;
    }

    /// Backward pass
    pub fn backward(self: *Self) !void {
        _ = self;
        // Full implementation would backprop through all layers
    }

    /// Collect all parameters
    pub fn collectParameters(self: *Self, allocator: std.mem.Allocator) ![]*Tensor {
        var params = std.ArrayList(*Tensor).init(allocator);
        errdefer params.deinit();

        // Embedding weights
        try params.append(self.embed_tokens.weight);

        // Block parameters
        for (self.blocks) |block| {
            try params.append(block.ln1.weight);
            try params.append(block.efla.w_k);
            try params.append(block.efla.w_v);
            try params.append(block.efla.w_o);
            if (block.efla.beta_param) |bp| {
                try params.append(bp);
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

        // Final norm and LM head
        try params.append(self.final_norm.weight);
        try params.append(self.lm_head.weight);

        return params.toOwnedSlice();
    }

    /// Get parameter names
    pub fn getParameterNames(self: *Self, allocator: std.mem.Allocator) ![]const []const u8 {
        var names = std.ArrayList([]const u8).init(allocator);
        errdefer {
            for (names.items) |n| allocator.free(n);
            names.deinit();
        }

        try names.append(try allocator.dupe(u8, "embed_tokens.weight"));

        for (self.blocks, 0..) |block, i| {
            try names.append(try std.fmt.allocPrint(allocator, "blocks.{d}.ln1.weight", .{i}));
            try names.append(try std.fmt.allocPrint(allocator, "blocks.{d}.efla.w_k", .{i}));
            try names.append(try std.fmt.allocPrint(allocator, "blocks.{d}.efla.w_v", .{i}));
            try names.append(try std.fmt.allocPrint(allocator, "blocks.{d}.efla.w_o", .{i}));
            for (0..block.prism.w_beta.len) |j| {
                try names.append(try std.fmt.allocPrint(allocator, "blocks.{d}.prism.w_beta.{d}", .{ i, j }));
            }
            try names.append(try std.fmt.allocPrint(allocator, "blocks.{d}.ln2.weight", .{i}));
            try names.append(try std.fmt.allocPrint(allocator, "blocks.{d}.mlp_up.weight", .{i}));
            try names.append(try std.fmt.allocPrint(allocator, "blocks.{d}.mlp_down.weight", .{i}));
        }

        try names.append(try allocator.dupe(u8, "final_norm.weight"));
        try names.append(try allocator.dupe(u8, "lm_head.weight"));

        return names.toOwnedSlice();
    }

    /// Collect gradients
    pub fn collectGradients(self: *Self, allocator: std.mem.Allocator) ![]*Tensor {
        var grads = std.ArrayList(*Tensor).init(allocator);
        errdefer grads.deinit();

        // Full implementation would collect all gradients
        // For now, placeholder
        _ = self;

        return grads.toOwnedSlice();
    }

    /// Count parameters
    pub fn countParameters(self: *Self) u64 {
        var count: u64 = 0;

        count += @as(u64, self.config.vocab_size) * self.config.hidden_dim;

        for (self.blocks) |_| {
            // Rough estimate per block
            count += @as(u64, self.config.hidden_dim) * self.config.hidden_dim * 4; // Attention
            count += @as(u64, self.config.hidden_dim) * self.config.intermediate_dim * 2; // MLP
            count += @as(u64, self.config.hidden_dim) * 2; // Norms
        }

        count += self.config.hidden_dim; // Final norm
        count += @as(u64, self.config.vocab_size) * self.config.hidden_dim; // LM head

        return count;
    }
};

test "EflaModel parameter count" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var prng = std.Random.DefaultPrng.init(42);
    var rng = prng.random();

    var config = config_mod.ModelConfig.default1T(gpa.allocator());
    // Use smaller config for test
    config.hidden_dim = 256;
    config.num_layers = 2;
    config.intermediate_dim = 512;
    config.num_heads = 4;

    var model = try EflaModel.init(gpa.allocator(), config, .cpu, 0, &rng);
    defer model.deinit();

    const param_count = model.countParameters();
    try std.testing.expect(param_count > 0);
}
