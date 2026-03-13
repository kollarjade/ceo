const std = @import("std");
const tensor_mod = @import("../tensor/tensor.zig");
const dtype_mod = @import("../tensor/dtype.zig");
const config_mod = @import("../util/config.zig");
const nn_mod = @import("../nn/layers.zig");
const efla_mod = @import("efla.zig");
const prism_mod = @import("prism.zig");

pub const Tensor = tensor_mod.Tensor;
pub const BF16 = dtype_mod.BF16;

pub const ModelError = error{
    InvalidConfiguration,
    ShapeMismatch,
    DTypeMismatch,
    DeviceMismatch,
    UnsupportedDevice,
    InvalidInputRank,
    UnsupportedOperation,
    GradientNotAvailable,
    BackwardNotReady,
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
        for (self.efla_states) |state| {
            if (state) |s| {
                s.deinit();
            }
        }
        for (self.prism_states) |state| {
            if (state) |s| {
                s.deinit();
            }
        }
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
    cached_input: ?*Tensor,
    cached_normed1: ?*Tensor,
    cached_efla_output: ?*Tensor,
    cached_prism_output: ?*Tensor,
    cached_residual1: ?*Tensor,
    cached_normed2: ?*Tensor,
    cached_mlp_up: ?*Tensor,
    cached_mlp_activated: ?*Tensor,
    cached_mlp_down: ?*Tensor,

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
        if (config.hidden_dim != config.num_heads * config.head_dim) {
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
            .cached_input = null,
            .cached_normed1 = null,
            .cached_efla_output = null,
            .cached_prism_output = null,
            .cached_residual1 = null,
            .cached_normed2 = null,
            .cached_mlp_up = null,
            .cached_mlp_activated = null,
            .cached_mlp_down = null,
        };

        return self;
    }

    pub fn deinit(self: *Self) void {
        self.clearCache();
        self.ln1.deinit();
        self.efla.deinit();
        self.prism.deinit();
        self.ln2.deinit();
        self.mlp_up.deinit();
        self.mlp_down.deinit();
        self.allocator.destroy(self);
    }

    fn clearCache(self: *Self) void {
        if (self.cached_input) |t| {
            t.deinit();
            self.cached_input = null;
        }
        if (self.cached_normed1) |t| {
            t.deinit();
            self.cached_normed1 = null;
        }
        if (self.cached_efla_output) |t| {
            t.deinit();
            self.cached_efla_output = null;
        }
        if (self.cached_prism_output) |t| {
            t.deinit();
            self.cached_prism_output = null;
        }
        if (self.cached_residual1) |t| {
            t.deinit();
            self.cached_residual1 = null;
        }
        if (self.cached_normed2) |t| {
            t.deinit();
            self.cached_normed2 = null;
        }
        if (self.cached_mlp_up) |t| {
            t.deinit();
            self.cached_mlp_up = null;
        }
        if (self.cached_mlp_activated) |t| {
            t.deinit();
            self.cached_mlp_activated = null;
        }
        if (self.cached_mlp_down) |t| {
            t.deinit();
            self.cached_mlp_down = null;
        }
    }

    pub fn forward(
        self: *Self,
        input: *Tensor,
        efla_state: ?*efla_mod.EflaState,
        prism_state: ?*prism_mod.PrismState,
    ) !TransformerBlockForwardResult {
        try self.validateTensorForBlock(input);
        self.clearCache();

        self.cached_input = try cloneTensor(self.allocator, input);
        errdefer {
            self.clearCache();
        }

        const normed = try self.ln1.forward(input);
        self.cached_normed1 = try cloneTensor(self.allocator, normed);

        const efla_result = try self.efla.forward(normed, efla_state);
        var new_efla_state = efla_result.new_state;
        self.cached_efla_output = try cloneTensor(self.allocator, efla_result.output);
        errdefer {
            if (new_efla_state) |s| {
                s.deinit();
            }
        }

        const prism_result = try self.prism.forward(normed, efla_result.output, prism_state);
        var new_prism_state = prism_result.new_state;
        self.cached_prism_output = try cloneTensor(self.allocator, prism_result.output);
        errdefer {
            if (new_prism_state) |s| {
                s.deinit();
            }
        }

        normed.deinit();

        const residual = try addTensorsFast(self.allocator, input, prism_result.output, self.device, self.device_id);
        self.cached_residual1 = try cloneTensor(self.allocator, residual);

        prism_result.output.deinit();
        efla_result.output.deinit();

        const normed2 = try self.ln2.forward(residual);
        self.cached_normed2 = try cloneTensor(self.allocator, normed2);

        const up = try self.mlp_up.forward(normed2);
        self.cached_mlp_up = try cloneTensor(self.allocator, up);
        normed2.deinit();

        const activated = try self.activation.forward(self.allocator, up);
        self.cached_mlp_activated = try cloneTensor(self.allocator, activated);
        up.deinit();

        const down = try self.mlp_down.forward(activated);
        self.cached_mlp_down = try cloneTensor(self.allocator, down);
        activated.deinit();

        const output = try addTensorsFast(self.allocator, residual, down, self.device, self.device_id);
        residual.deinit();
        down.deinit();

        return .{
            .output = output,
            .new_efla_state = new_efla_state,
            .new_prism_state = new_prism_state,
        };
    }

    fn validateTensorForBlock(self: *Self, tensor: *Tensor) !void {
        if (tensor.dtype != .bf16) {
            return ModelError.DTypeMismatch;
        }
        if (tensor.device != self.device or tensor.device_id != self.device_id) {
            return ModelError.DeviceMismatch;
        }
        if (tensor.shape.ndim != 3) {
            return ModelError.InvalidInputRank;
        }
        if (tensor.shape.dim(2) != self.config.hidden_dim) {
            return ModelError.ShapeMismatch;
        }
    }

    pub fn backward(self: *Self, grad_output: *Tensor) !*Tensor {
        if (self.cached_input == null or self.cached_residual1 == null) {
            return ModelError.BackwardNotReady;
        }

        const grad_residual1 = try cloneTensor(self.allocator, grad_output);
        defer grad_residual1.deinit();

        const grad_down = try self.mlp_down.backward(grad_output);
        defer grad_down.deinit();

        const grad_activated = try geluBackward(
            self.allocator,
            grad_down,
            self.cached_mlp_up orelse return ModelError.BackwardNotReady,
            self.device,
            self.device_id,
        );
        defer grad_activated.deinit();

        const grad_up = try self.mlp_up.backward(grad_activated);
        defer grad_up.deinit();

        const grad_ln2 = try self.ln2.backward(grad_up);
        defer grad_ln2.deinit();

        const grad_after_first_residual = try addTensorsFast(
            self.allocator,
            grad_residual1,
            grad_ln2,
            self.device,
            self.device_id,
        );
        defer grad_after_first_residual.deinit();

        const grad_prism = try self.prism.backward(grad_after_first_residual);
        defer grad_prism.deinit();

        const grad_efla = try self.efla.backward(grad_prism);
        defer grad_efla.deinit();

        const grad_ln1 = try self.ln1.backward(grad_efla);
        defer grad_ln1.deinit();

        const grad_input = try addTensorsFast(
            self.allocator,
            grad_after_first_residual,
            grad_ln1,
            self.device,
            self.device_id,
        );

        return grad_input;
    }
};

pub const EflaModel = struct {
    embed_tokens: *nn_mod.Embedding,
    blocks: []*TransformerBlock,
    final_norm: *nn_mod.RMSNorm,
    lm_head: ?*nn_mod.Linear,
    config: config_mod.ModelConfig,
    allocator: std.mem.Allocator,
    device: tensor_mod.Device,
    device_id: i32,
    tied_embeddings: bool,
    cached_embeds: ?*Tensor,
    cached_hidden_states: ?[]*Tensor,
    cached_final_normed: ?*Tensor,

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
        if (config.hidden_dim != config.num_heads * config.head_dim) {
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

        var lm_head: ?*nn_mod.Linear = null;
        errdefer {
            if (lm_head) |head| {
                head.deinit();
            }
        }

        if (!config.tie_embeddings) {
            lm_head = try nn_mod.Linear.init(
                allocator,
                config.hidden_dim,
                config.vocab_size,
                false,
                device,
                device_id,
                rng,
            );
        }

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
            .cached_embeds = null,
            .cached_hidden_states = null,
            .cached_final_normed = null,
        };

        return self;
    }

    pub fn deinit(self: *Self) void {
        self.clearCache();
        self.embed_tokens.deinit();
        for (self.blocks) |block| {
            block.deinit();
        }
        self.allocator.free(self.blocks);
        self.final_norm.deinit();
        if (self.lm_head) |head| {
            head.deinit();
        }
        self.allocator.destroy(self);
    }

    fn clearCache(self: *Self) void {
        if (self.cached_embeds) |t| {
            t.deinit();
            self.cached_embeds = null;
        }
        if (self.cached_hidden_states) |states| {
            for (states) |t| {
                t.deinit();
            }
            self.allocator.free(states);
            self.cached_hidden_states = null;
        }
        if (self.cached_final_normed) |t| {
            t.deinit();
            self.cached_final_normed = null;
        }
    }

    pub fn forward(self: *Self, input_ids: *Tensor) !*Tensor {
        var result = try self.forwardWithStates(input_ids, null, null);

        for (result.efla_states) |state| {
            if (state) |s| {
                s.deinit();
            }
        }
        for (result.prism_states) |state| {
            if (state) |s| {
                s.deinit();
            }
        }
        self.allocator.free(result.efla_states);
        self.allocator.free(result.prism_states);

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
        if (input_ids.shape.ndim != 2) {
            return ModelError.InvalidInputRank;
        }

        self.clearCache();

        const embeds = try self.embed_tokens.forward(input_ids);
        self.cached_embeds = try cloneTensor(self.allocator, embeds);
        errdefer self.clearCache();

        var hidden: *Tensor = embeds;

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

        const hidden_states_cache = try self.allocator.alloc(*Tensor, self.blocks.len);
        errdefer {
            for (hidden_states_cache[0..self.blocks.len]) |t| {
                _ = t;
            }
            self.allocator.free(hidden_states_cache);
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

            hidden_states_cache[i] = try cloneTensor(self.allocator, hidden);

            const result = try block.forward(hidden, efla_state, prism_state);
            hidden.deinit();
            hidden = result.output;
            efla_states_out[i] = result.new_efla_state;
            prism_states_out[i] = result.new_prism_state;
        }

        self.cached_hidden_states = hidden_states_cache;

        const normed = try self.final_norm.forward(hidden);
        self.cached_final_normed = try cloneTensor(self.allocator, normed);
        hidden.deinit();
        errdefer normed.deinit();

        const logits = try self.projectToVocab(normed);
        normed.deinit();

        return .{
            .logits = logits,
            .efla_states = efla_states_out,
            .prism_states = prism_states_out,
        };
    }

    pub fn backward(self: *Self, grad_output: *Tensor) !*Tensor {
        if (self.cached_final_normed == null) {
            return ModelError.BackwardNotReady;
        }

        var grad_hidden: *Tensor = undefined;

        if (!self.tied_embeddings) {
            const head = self.lm_head orelse return ModelError.InvalidConfiguration;
            grad_hidden = try head.backward(grad_output);
        } else {
            grad_hidden = try vocabProjectBackward(
                self.allocator,
                grad_output,
                self.embed_tokens.weight,
                self.config,
                self.device,
                self.device_id,
            );
        }
        errdefer grad_hidden.deinit();

        const grad_after_norm = try self.final_norm.backward(grad_hidden);
        grad_hidden.deinit();
        grad_hidden = grad_after_norm;

        var i: usize = self.blocks.len;
        while (i > 0) {
            i -= 1;
            const grad_block = try self.blocks[i].backward(grad_hidden);
            grad_hidden.deinit();
            grad_hidden = grad_block;
        }

        const grad_embed = try self.embed_tokens.backward(grad_hidden);
        grad_hidden.deinit();

        return grad_embed;
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
        if (self.lm_head) |head| {
            try params.append(head.weight);
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
        if (self.lm_head != null) {
            try names.append(try allocator.dupe(u8, "lm_head.weight"));
        }

        return names.toOwnedSlice();
    }

    pub fn collectGradients(self: *Self, allocator: std.mem.Allocator) ![]*Tensor {
        var grads = std.ArrayList(*Tensor).init(allocator);
        errdefer grads.deinit();

        const embed_grad = self.embed_tokens.weight.grad orelse return ModelError.GradientNotAvailable;
        try grads.append(embed_grad);

        for (self.blocks) |block| {
            const ln1_grad = block.ln1.weight.grad orelse return ModelError.GradientNotAvailable;
            try grads.append(ln1_grad);

            const efla_wk_grad = block.efla.w_k.grad orelse return ModelError.GradientNotAvailable;
            try grads.append(efla_wk_grad);

            const efla_wv_grad = block.efla.w_v.grad orelse return ModelError.GradientNotAvailable;
            try grads.append(efla_wv_grad);

            const efla_wo_grad = block.efla.w_o.grad orelse return ModelError.GradientNotAvailable;
            try grads.append(efla_wo_grad);

            if (block.efla.beta_param) |beta_param| {
                const beta_grad = beta_param.grad orelse return ModelError.GradientNotAvailable;
                try grads.append(beta_grad);
            }

            for (block.prism.w_beta) |w| {
                const w_grad = w.grad orelse return ModelError.GradientNotAvailable;
                try grads.append(w_grad);
            }
            for (block.prism.w_k) |w| {
                const w_grad = w.grad orelse return ModelError.GradientNotAvailable;
                try grads.append(w_grad);
            }
            for (block.prism.w_p) |w| {
                const w_grad = w.grad orelse return ModelError.GradientNotAvailable;
                try grads.append(w_grad);
            }

            const sc_grad = block.prism.shortconv.weight.grad orelse return ModelError.GradientNotAvailable;
            try grads.append(sc_grad);

            const ln2_grad = block.ln2.weight.grad orelse return ModelError.GradientNotAvailable;
            try grads.append(ln2_grad);

            const mlp_up_grad = block.mlp_up.weight.grad orelse return ModelError.GradientNotAvailable;
            try grads.append(mlp_up_grad);

            const mlp_down_grad = block.mlp_down.weight.grad orelse return ModelError.GradientNotAvailable;
            try grads.append(mlp_down_grad);
        }

        const final_norm_grad = self.final_norm.weight.grad orelse return ModelError.GradientNotAvailable;
        try grads.append(final_norm_grad);

        if (self.lm_head) |head| {
            const head_grad = head.weight.grad orelse return ModelError.GradientNotAvailable;
            try grads.append(head_grad);
        }

        return grads.toOwnedSlice();
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
        if (self.lm_head) |head| {
            count += tensorNumel(head.weight);
        }

        return count;
    }

    fn projectToVocab(self: *Self, hidden: *Tensor) !*Tensor {
        if (!self.tied_embeddings) {
            const head = self.lm_head orelse return ModelError.InvalidConfiguration;
            return head.forward(hidden);
        }

        if (hidden.device != self.device or hidden.device_id != self.device_id) {
            return ModelError.DeviceMismatch;
        }
        if (hidden.dtype != .bf16 or self.embed_tokens.weight.dtype != .bf16) {
            return ModelError.DTypeMismatch;
        }
        if (hidden.shape.ndim != 2 and hidden.shape.ndim != 3) {
            return ModelError.InvalidInputRank;
        }
        if (self.embed_tokens.weight.shape.ndim != 2) {
            return ModelError.InvalidConfiguration;
        }
        if (hidden.shape.dim(hidden.shape.ndim - 1) != self.embed_tokens.weight.shape.dim(1)) {
            return ModelError.ShapeMismatch;
        }
        if (self.embed_tokens.weight.shape.dim(0) != self.config.vocab_size) {
            return ModelError.ShapeMismatch;
        }

        const output_shape = switch (hidden.shape.ndim) {
            2 => tensor_mod.Shape.init(&[_]usize{ hidden.shape.dim(0), self.config.vocab_size }),
            3 => tensor_mod.Shape.init(&[_]usize{ hidden.shape.dim(0), hidden.shape.dim(1), self.config.vocab_size }),
            else => return ModelError.InvalidInputRank,
        };

        const output = try Tensor.init(self.allocator, output_shape, .bf16, self.device, self.device_id);
        errdefer output.deinit();

        const hidden_ptr = hidden.typedPtr(BF16) orelse return ModelError.UnsupportedOperation;
        const weight_ptr = self.embed_tokens.weight.typedPtr(BF16) orelse return ModelError.UnsupportedOperation;
        const output_ptr = output.typedPtr(BF16) orelse return ModelError.UnsupportedOperation;

        const hidden_dim = self.config.hidden_dim;
        const vocab_size = self.config.vocab_size;

        switch (hidden.shape.ndim) {
            2 => {
                const rows = hidden.shape.dim(0);
                for (0..rows) |row| {
                    matVecRowBF16(
                        output_ptr[row * vocab_size ..][0..vocab_size],
                        hidden_ptr[row * hidden_dim ..][0..hidden_dim],
                        weight_ptr,
                        vocab_size,
                        hidden_dim,
                    );
                }
            },
            3 => {
                const batch_size = hidden.shape.dim(0);
                const seq_len = hidden.shape.dim(1);
                for (0..batch_size) |batch_idx| {
                    for (0..seq_len) |token_idx| {
                        const row_offset = (batch_idx * seq_len + token_idx) * hidden_dim;
                        const out_offset = (batch_idx * seq_len + token_idx) * vocab_size;
                        matVecRowBF16(
                            output_ptr[out_offset..][0..vocab_size],
                            hidden_ptr[row_offset..][0..hidden_dim],
                            weight_ptr,
                            vocab_size,
                            hidden_dim,
                        );
                    }
                }
            },
            else => return ModelError.InvalidInputRank,
        }

        return output;
    }
};

fn matVecRowBF16(
    output: []BF16,
    input: []const BF16,
    weight: []const BF16,
    num_rows: usize,
    num_cols: usize,
) void {
    const VecSize = 8;
    const aligned_cols = (num_cols / VecSize) * VecSize;

    for (0..num_rows) |row| {
        var sum_vec: @Vector(VecSize, f32) = @splat(0.0);
        const weight_row = weight[row * num_cols ..][0..num_cols];

        var k: usize = 0;
        while (k < aligned_cols) : (k += VecSize) {
            var input_vec: @Vector(VecSize, f32) = undefined;
            var weight_vec: @Vector(VecSize, f32) = undefined;
            inline for (0..VecSize) |vi| {
                input_vec[vi] = input[k + vi].toFloat32();
                weight_vec[vi] = weight_row[k + vi].toFloat32();
            }
            sum_vec += input_vec * weight_vec;
        }

        var sum: f32 = @reduce(.Add, sum_vec);

        while (k < num_cols) : (k += 1) {
            sum += input[k].toFloat32() * weight_row[k].toFloat32();
        }

        output[row] = BF16.fromFloat32(sum);
    }
}

fn addTensorsFast(
    allocator: std.mem.Allocator,
    a: *Tensor,
    b: *Tensor,
    device: tensor_mod.Device,
    device_id: i32,
) !*Tensor {
    if (a.dtype != .bf16 or b.dtype != .bf16) {
        return ModelError.DTypeMismatch;
    }
    if (a.device != device or a.device_id != device_id) {
        return ModelError.DeviceMismatch;
    }
    if (b.device != device or b.device_id != device_id) {
        return ModelError.DeviceMismatch;
    }
    if (!a.shape.equalTo(b.shape)) {
        return ModelError.ShapeMismatch;
    }

    const a_ptr = a.typedPtr(BF16) orelse return ModelError.UnsupportedOperation;
    const b_ptr = b.typedPtr(BF16) orelse return ModelError.UnsupportedOperation;

    const output = try Tensor.init(allocator, a.shape, a.dtype, device, device_id);
    errdefer output.deinit();

    const o_ptr = output.typedPtr(BF16) orelse return ModelError.UnsupportedOperation;
    const numel = a.shape.numel();

    const VecSize = 8;
    const aligned = (numel / VecSize) * VecSize;

    var i: usize = 0;
    while (i < aligned) : (i += VecSize) {
        var a_vec: @Vector(VecSize, f32) = undefined;
        var b_vec: @Vector(VecSize, f32) = undefined;
        inline for (0..VecSize) |vi| {
            a_vec[vi] = a_ptr[i + vi].toFloat32();
            b_vec[vi] = b_ptr[i + vi].toFloat32();
        }
        const result_vec = a_vec + b_vec;
        inline for (0..VecSize) |vi| {
            o_ptr[i + vi] = BF16.fromFloat32(result_vec[vi]);
        }
    }

    while (i < numel) : (i += 1) {
        o_ptr[i] = BF16.fromFloat32(a_ptr[i].toFloat32() + b_ptr[i].toFloat32());
    }

    return output;
}

fn cloneTensor(allocator: std.mem.Allocator, src: *Tensor) !*Tensor {
    const dst = try Tensor.init(allocator, src.shape, src.dtype, src.device, src.device_id);
    errdefer dst.deinit();

    const src_bytes = src.rawBytes() orelse return ModelError.UnsupportedOperation;
    const dst_bytes = dst.rawMutBytes() orelse return ModelError.UnsupportedOperation;

    if (src_bytes.len != dst_bytes.len) {
        return ModelError.ShapeMismatch;
    }

    @memcpy(dst_bytes, src_bytes);

    return dst;
}

fn geluBackward(
    allocator: std.mem.Allocator,
    grad_output: *Tensor,
    input: *Tensor,
    device: tensor_mod.Device,
    device_id: i32,
) !*Tensor {
    if (grad_output.dtype != .bf16 or input.dtype != .bf16) {
        return ModelError.DTypeMismatch;
    }
    if (!grad_output.shape.equalTo(input.shape)) {
        return ModelError.ShapeMismatch;
    }

    const output = try Tensor.init(allocator, grad_output.shape, .bf16, device, device_id);
    errdefer output.deinit();

    const grad_ptr = grad_output.typedPtr(BF16) orelse return ModelError.UnsupportedOperation;
    const input_ptr = input.typedPtr(BF16) orelse return ModelError.UnsupportedOperation;
    const out_ptr = output.typedPtr(BF16) orelse return ModelError.UnsupportedOperation;

    const numel = grad_output.shape.numel();
    const sqrt_2_over_pi: f32 = 0.7978845608028654;
    const coeff: f32 = 0.044715;

    for (0..numel) |i| {
        const x = input_ptr[i].toFloat32();
        const x3 = x * x * x;
        const inner = sqrt_2_over_pi * (x + coeff * x3);
        const tanh_inner = std.math.tanh(inner);
        const sech2 = 1.0 - tanh_inner * tanh_inner;
        const gelu_grad = 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * sqrt_2_over_pi * (1.0 + 3.0 * coeff * x * x);
        out_ptr[i] = BF16.fromFloat32(grad_ptr[i].toFloat32() * gelu_grad);
    }

    return output;
}

fn vocabProjectBackward(
    allocator: std.mem.Allocator,
    grad_output: *Tensor,
    embed_weight: *Tensor,
    config: config_mod.ModelConfig,
    device: tensor_mod.Device,
    device_id: i32,
) !*Tensor {
    if (grad_output.dtype != .bf16 or embed_weight.dtype != .bf16) {
        return ModelError.DTypeMismatch;
    }

    const last_dim = grad_output.shape.dim(grad_output.shape.ndim - 1);
    if (last_dim != config.vocab_size) {
        return ModelError.ShapeMismatch;
    }

    const hidden_dim = config.hidden_dim;
    const vocab_size = config.vocab_size;

    var output_shape_dims: [3]usize = undefined;
    var output_ndim: usize = undefined;
    switch (grad_output.shape.ndim) {
        2 => {
            output_shape_dims[0] = grad_output.shape.dim(0);
            output_shape_dims[1] = hidden_dim;
            output_ndim = 2;
        },
        3 => {
            output_shape_dims[0] = grad_output.shape.dim(0);
            output_shape_dims[1] = grad_output.shape.dim(1);
            output_shape_dims[2] = hidden_dim;
            output_ndim = 3;
        },
        else => return ModelError.InvalidInputRank,
    }

    const output_shape = tensor_mod.Shape.init(output_shape_dims[0..output_ndim]);
    const output = try Tensor.init(allocator, output_shape, .bf16, device, device_id);
    errdefer output.deinit();

    const grad_ptr = grad_output.typedPtr(BF16) orelse return ModelError.UnsupportedOperation;
    const weight_ptr = embed_weight.typedPtr(BF16) orelse return ModelError.UnsupportedOperation;
    const out_ptr = output.typedPtr(BF16) orelse return ModelError.UnsupportedOperation;

    const total_tokens = output.shape.numel() / hidden_dim;

    for (0..total_tokens) |token| {
        const grad_offset = token * vocab_size;
        const out_offset = token * hidden_dim;

        for (0..hidden_dim) |h| {
            var sum: f32 = 0.0;
            for (0..vocab_size) |v| {
                sum += grad_ptr[grad_offset + v].toFloat32() * weight_ptr[v * hidden_dim + h].toFloat32();
            }
            out_ptr[out_offset + h] = BF16.fromFloat32(sum);
        }
    }

    return output;
}

fn tensorNumel(tensor: *Tensor) u64 {
    return @intCast(tensor.shape.numel());
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
        manual_count += @intCast(param.shape.numel());
    }

    try std.testing.expectEqual(manual_count, model.countParameters());
}
