const std = @import("std");
const cli = @import("util/cli.zig");
const config = @import("util/config.zig");
const runtime = @import("runtime/main.zig");
const trainer = @import("runtime/trainer.zig");
const tokenizer_mod = @import("data/tokenizer.zig");
const dataset = @import("data/dataset.zig");
const checkpoint = @import("checkpoint/manager.zig");
const evaluator = @import("eval/evaluator.zig");
const telemetry = @import("telemetry/main.zig");

pub const std_options = std.Options{
    .logFn = telemetry.logFn,
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{
        .enable_memory_limit = true,
        .enable_thread_safety = true,
    }){};
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();

    // Parse command line arguments
    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();

    _ = args.next(); // Skip program name

    const command_str = args.next() orelse {
        try printUsage();
        return error.MissingCommand;
    };

    const command = std.meta.stringToEnum(Command, command_str) orelse {
        std.log.err("Unknown command: {s}", .{command_str});
        try printUsage();
        return error.InvalidCommand;
    };

    switch (command) {
        .train => try runTrain(allocator, &args),
        .evaluate => try runEvaluate(allocator, &args),
        .generate => try runGenerate(allocator, &args),
        .@"smoke-test" => try runSmokeTest(allocator, &args),
        .checkpoint => try runCheckpoint(allocator, &args),
        .tokenizer => try runTokenizer(allocator, &args),
        .validate => try runValidate(allocator, &args),
        .profile => try runProfile(allocator, &args),
        .version => try printVersion(),
        .help => try printUsage(),
    }
}

const Command = enum {
    train,
    evaluate,
    generate,
    @"smoke-test",
    checkpoint,
    tokenizer,
    validate,
    profile,
    version,
    help,
};

fn printUsage() !void {
    const stdout = std.io.getStdOut().writer();
    try stdout.print(
        \\EFLA Trainer - Ultra-long context LLM training system
        \\
        \\Usage: efla-train <command> [options]
        \\
        \\Commands:
        \\  train       Run distributed training
        \\  evaluate    Evaluate model perplexity
        \\  generate    Generate text from model
        \\  smoke-test  Run smoke test on single GPU
        \\  checkpoint  Manage checkpoints (list, validate, convert)
        \\  tokenizer   Train or use tokenizer
        \\  validate    Validate configuration and model
        \\  profile     Run with profiling hooks
        \\  version     Print version information
        \\  help        Print this help message
        \\
        \\Examples:
        \\  efla-train train --config configs/train.yaml
        \\  efla-train evaluate --checkpoint path/to/checkpoint --data path/to/eval.bin
        \\  efla-train generate --checkpoint path/to/checkpoint --prompt "Hello"
        \\  efla-train smoke-test --config configs/smoke.yaml
        \\  efla-train tokenizer train --corpus data.txt --vocab-size 65536
        \\
    , .{});
}

fn printVersion() !void {
    const stdout = std.io.getStdOut().writer();
    try stdout.print("EFLA Trainer v0.1.0\n", .{});
    try stdout.print("Built with Zig {s}\n", .{builtin.zig_version_string});
    try stdout.print("Target: NVIDIA Blackwell SM100 (8×B200)\n", .{});
}

const builtin = @import("builtin");

fn runTrain(allocator: std.mem.Allocator, args: *std.process.ArgIterator) !void {
    var config_path: ?[]const u8 = null;
    var resume_path: ?[]const u8 = null;
    var data_path: ?[]const u8 = null;
    var checkpoint_dir: ?[]const u8 = null;

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--config")) {
            config_path = args.next() orelse return error.MissingConfigPath;
        } else if (std.mem.eql(u8, arg, "--resume")) {
            resume_path = args.next() orelse return error.MissingResumePath;
        } else if (std.mem.eql(u8, arg, "--data")) {
            data_path = args.next() orelse return error.MissingDataPath;
        } else if (std.mem.eql(u8, arg, "--checkpoint-dir")) {
            checkpoint_dir = args.next() orelse return error.MissingCheckpointDir;
        } else {
            std.log.err("Unknown argument: {s}", .{arg});
            return error.InvalidArgument;
        }
    }

    const cfg_path = config_path orelse return error.MissingConfig;
    const cfg = try config.Config.parseFromFile(allocator, cfg_path);
    defer cfg.deinit(allocator);

    std.log.info("Starting training with config: {s}", .{cfg_path});

    // Initialize distributed runtime
    var dist_runtime = try runtime.DistributedRuntime.init(allocator, cfg.runtime);
    defer dist_runtime.deinit();

    std.log.info("Initialized distributed runtime: rank={d}, world_size={d}", .{
        dist_runtime.rank,
        dist_runtime.world_size,
    });

    // Initialize telemetry
    var tele = try telemetry.Telemetry.init(allocator, cfg.telemetry, dist_runtime.rank);
    defer tele.deinit();

    // Run training
    var train = try trainer.Trainer.init(allocator, cfg, &dist_runtime, &tele);
    defer train.deinit();

    if (resume_path) |path| {
        std.log.info("Resuming from checkpoint: {s}", .{path});
        try train.resume(path);
    }

    try train.run(data_path, checkpoint_dir);
}

fn runEvaluate(allocator: std.mem.Allocator, args: *std.process.ArgIterator) !void {
    var checkpoint_path: ?[]const u8 = null;
    var data_path: ?[]const u8 = null;
    var config_path: ?[]const u8 = null;
    var max_tokens: usize = std.math.maxInt(usize);

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--checkpoint")) {
            checkpoint_path = args.next() orelse return error.MissingCheckpointPath;
        } else if (std.mem.eql(u8, arg, "--data")) {
            data_path = args.next() orelse return error.MissingDataPath;
        } else if (std.mem.eql(u8, arg, "--config")) {
            config_path = args.next() orelse return error.MissingConfigPath;
        } else if (std.mem.eql(u8, arg, "--max-tokens")) {
            const max_str = args.next() orelse return error.MissingMaxTokens;
            max_tokens = try std.fmt.parseInt(usize, max_str, 10);
        }
    }

    const ckpt_path = checkpoint_path orelse return error.MissingCheckpoint;
    const data = data_path orelse return error.MissingDataPath;

    // Load config from checkpoint or provided path
    const cfg: config.Config = if (config_path) |path|
        try config.Config.parseFromFile(allocator, path)
    else
        try config.Config.parseFromFile(allocator, try std.fmt.bufPrint(try allocator.alloc(u8, 4096), "{s}/config.yaml", .{ckpt_path}));
    defer cfg.deinit(allocator);

    var eval = try evaluator.Evaluator.init(allocator, cfg, ckpt_path);
    defer eval.deinit();

    const results = try eval.evaluatePerplexity(data, max_tokens);
    std.log.info("Evaluation complete:", .{});
    std.log.info("  Perplexity: {d:.4}", .{results.perplexity});
    std.log.info("  Loss: {d:.4}", .{results.loss});
    std.log.info("  Tokens: {d}", .{results.tokens});
}

fn runGenerate(allocator: std.mem.Allocator, args: *std.process.ArgIterator) !void {
    var checkpoint_path: ?[]const u8 = null;
    var prompt: ?[]const u8 = null;
    var max_new_tokens: usize = 256;
    var temperature: f32 = 1.0;
    var top_p: f32 = 0.9;
    var config_path: ?[]const u8 = null;

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--checkpoint")) {
            checkpoint_path = args.next() orelse return error.MissingCheckpointPath;
        } else if (std.mem.eql(u8, arg, "--prompt")) {
            prompt = args.next() orelse return error.MissingPrompt;
        } else if (std.mem.eql(u8, arg, "--max-tokens")) {
            const max_str = args.next() orelse return error.MissingMaxTokens;
            max_new_tokens = try std.fmt.parseInt(usize, max_str, 10);
        } else if (std.mem.eql(u8, arg, "--temperature")) {
            const temp_str = args.next() orelse return error.MissingTemperature;
            temperature = try std.fmt.parseFloat(f32, temp_str);
        } else if (std.mem.eql(u8, arg, "--top-p")) {
            const top_p_str = args.next() orelse return error.MissingTopP;
            top_p = try std.fmt.parseFloat(f32, top_p_str);
        } else if (std.mem.eql(u8, arg, "--config")) {
            config_path = args.next() orelse return error.MissingConfigPath;
        }
    }

    const ckpt_path = checkpoint_path orelse return error.MissingCheckpoint;
    const prompt_text = prompt orelse "";

    // Load config
    const cfg: config.Config = if (config_path) |path|
        try config.Config.parseFromFile(allocator, path)
    else
        try config.Config.parseFromFile(allocator, try std.fmt.bufPrint(try allocator.alloc(u8, 4096), "{s}/config.yaml", .{ckpt_path}));
    defer cfg.deinit(allocator);

    var gen = try evaluator.Generator.init(allocator, cfg, ckpt_path);
    defer gen.deinit();

    const output = try gen.generate(prompt_text, max_new_tokens, temperature, top_p);

    const stdout = std.io.getStdOut().writer();
    try stdout.print("{s}\n", .{output});
}

fn runSmokeTest(allocator: std.mem.Allocator, args: *std.process.ArgIterator) !void {
    _ = args;

    std.log.info("Running smoke test...", .{});

    // Load smoke test config
    const cfg = try config.Config.parseFromFile(allocator, "configs/smoke.yaml");
    defer cfg.deinit(allocator);

    // Verify GPU availability
    var gpu_check = try runtime.cuda.initCUDA();
    defer gpu_check.deinit();

    if (gpu_check.device_count == 0) {
        std.log.err("No CUDA devices found. Smoke test requires at least 1 GPU.", .{});
        return error.NoGpuAvailable;
    }

    std.log.info("Found {d} CUDA device(s)", .{gpu_check.device_count});

    // Run minimal training iteration
    var dist_runtime = try runtime.DistributedRuntime.initSingleGPU(allocator);
    defer dist_runtime.deinit();

    var tele = try telemetry.Telemetry.init(allocator, cfg.telemetry, 0);
    defer tele.deinit();

    var train = try trainer.Trainer.init(allocator, cfg, &dist_runtime, &tele);
    defer train.deinit();

    try train.smokeTest();

    std.log.info("Smoke test passed!", .{});
}

fn runCheckpoint(allocator: std.mem.Allocator, args: *std.process.ArgIterator) !void {
    const subcommand = args.next() orelse return error.MissingSubcommand;

    if (std.mem.eql(u8, subcommand, "list")) {
        var dir_path: ?[]const u8 = null;

        while (args.next()) |arg| {
            if (std.mem.eql(u8, arg, "--dir")) {
                dir_path = args.next() orelse return error.MissingDirPath;
            }
        }

        const path = dir_path orelse "checkpoints";
        try checkpoint.listCheckpoints(allocator, path);
    } else if (std.mem.eql(u8, subcommand, "validate")) {
        var ckpt_path: ?[]const u8 = null;

        while (args.next()) |arg| {
            if (std.mem.eql(u8, arg, "--path")) {
                ckpt_path = args.next() orelse return error.MissingCheckpointPath;
            }
        }

        const cp_path = ckpt_path orelse return error.MissingCheckpoint;
        try checkpoint.validateCheckpoint(allocator, cp_path);
    } else if (std.mem.eql(u8, subcommand, "convert")) {
        var input_path: ?[]const u8 = null;
        var output_path: ?[]const u8 = null;
        var format: checkpoint.CheckpointFormat = .efla_native;

        while (args.next()) |arg| {
            if (std.mem.eql(u8, arg, "--input")) {
                input_path = args.next() orelse return error.MissingInputPath;
            } else if (std.mem.eql(u8, arg, "--output")) {
                output_path = args.next() orelse return error.MissingOutputPath;
            } else if (std.mem.eql(u8, arg, "--format")) {
                const format_str = args.next() orelse return error.MissingFormat;
                format = std.meta.stringToEnum(checkpoint.CheckpointFormat, format_str) orelse return error.InvalidFormat;
            }
        }

        const in_path = input_path orelse return error.MissingInput;
        const out_path = output_path orelse return error.MissingOutput;

        try checkpoint.convertCheckpoint(allocator, in_path, out_path, format);
    } else {
        std.log.err("Unknown checkpoint subcommand: {s}", .{subcommand});
        return error.InvalidSubcommand;
    }
}

fn runTokenizer(allocator: std.mem.Allocator, args: *std.process.ArgIterator) !void {
    const subcommand = args.next() orelse return error.MissingSubcommand;

    if (std.mem.eql(u8, subcommand, "train")) {
        var corpus_path: ?[]const u8 = null;
        var output_path: ?[]const u8 = null;
        var vocab_size: usize = 65536;
        var model_type: tokenizer_mod.TokenizerType = .bpe;

        while (args.next()) |arg| {
            if (std.mem.eql(u8, arg, "--corpus")) {
                corpus_path = args.next() orelse return error.MissingCorpusPath;
            } else if (std.mem.eql(u8, arg, "--output")) {
                output_path = args.next() orelse return error.MissingOutputPath;
            } else if (std.mem.eql(u8, arg, "--vocab-size")) {
                const vs_str = args.next() orelse return error.MissingVocabSize;
                vocab_size = try std.fmt.parseInt(usize, vs_str, 10);
            } else if (std.mem.eql(u8, arg, "--type")) {
                const type_str = args.next() orelse return error.MissingType;
                model_type = std.meta.stringToEnum(tokenizer_mod.TokenizerType, type_str) orelse return error.InvalidType;
            }
        }

        const corpus = corpus_path orelse return error.MissingCorpus;
        const output = output_path orelse "tokenizer.bin";

        var tok = try tokenizer_mod.Tokenizer.train(allocator, corpus, vocab_size, model_type);
        defer tok.deinit();

        try tok.save(output);
        std.log.info("Tokenizer saved to {s}", .{output});
    } else if (std.mem.eql(u8, subcommand, "encode")) {
        var tokenizer_path: ?[]const u8 = null;
        var input_text: ?[]const u8 = null;
        var input_file: ?[]const u8 = null;
        var output_file: ?[]const u8 = null;

        while (args.next()) |arg| {
            if (std.mem.eql(u8, arg, "--tokenizer")) {
                tokenizer_path = args.next() orelse return error.MissingTokenizerPath;
            } else if (std.mem.eql(u8, arg, "--text")) {
                input_text = args.next() orelse return error.MissingText;
            } else if (std.mem.eql(u8, arg, "--input")) {
                input_file = args.next() orelse return error.MissingInputFile;
            } else if (std.mem.eql(u8, arg, "--output")) {
                output_file = args.next() orelse return error.MissingOutputFile;
            }
        }

        const tok_path = tokenizer_path orelse return error.MissingTokenizer;
        var tok = try tokenizer_mod.Tokenizer.load(allocator, tok_path);
        defer tok.deinit();

        if (input_file) |in_file| {
            const out_file = output_file orelse return error.MissingOutput;
            try tok.encodeFile(in_file, out_file);
        } else if (input_text) |text| {
            const tokens = try tok.encode(allocator, text);
            defer allocator.free(tokens);

            const stdout = std.io.getStdOut().writer();
            for (tokens) |t| {
                try stdout.print("{d} ", .{t});
            }
            try stdout.print("\n", .{});
        }
    } else if (std.mem.eql(u8, subcommand, "decode")) {
        var tokenizer_path: ?[]const u8 = null;
        var tokens: ?[]const u32 = null;

        while (args.next()) |arg| {
            if (std.mem.eql(u8, arg, "--tokenizer")) {
                tokenizer_path = args.next() orelse return error.MissingTokenizerPath;
            } else if (std.mem.eql(u8, arg, "--tokens")) {
                const tokens_str = args.next() orelse return error.MissingTokens;
                var list = std.ArrayList(u32).init(allocator);
                defer list.deinit();

                var iter = std.mem.split(u8, tokens_str, ",");
                while (iter.next()) |t| {
                    const trimmed = std.mem.trim(u8, t, " \t\n");
                    if (trimmed.len > 0) {
                        try list.append(try std.fmt.parseInt(u32, trimmed, 10));
                    }
                }
                tokens = try list.toOwnedSlice();
            }
        }

        const tok_path = tokenizer_path orelse return error.MissingTokenizer;
        const token_list = tokens orelse return error.MissingTokens;

        var tok = try tokenizer_mod.Tokenizer.load(allocator, tok_path);
        defer tok.deinit();

        const text = try tok.decode(allocator, token_list);
        defer allocator.free(text);

        const stdout = std.io.getStdOut().writer();
        try stdout.print("{s}\n", .{text});
    } else {
        std.log.err("Unknown tokenizer subcommand: {s}", .{subcommand});
        return error.InvalidSubcommand;
    }
}

fn runValidate(allocator: std.mem.Allocator, args: *std.process.ArgIterator) !void {
    var config_path: ?[]const u8 = null;

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--config")) {
            config_path = args.next() orelse return error.MissingConfigPath;
        }
    }

    const cfg_path = config_path orelse return error.MissingConfig;
    const cfg = try config.Config.parseFromFile(allocator, cfg_path);
    defer cfg.deinit(allocator);

    std.log.info("Validating configuration...", .{});

    // Validate model configuration
    try cfg.model.validate();

    // Validate training configuration
    try cfg.training.validate();

    // Validate memory requirements
    const mem_estimate = try cfg.model.estimateMemory();
    std.log.info("Estimated memory per GPU: {d:.2} GB", .{@as(f64, @floatFromInt(mem_estimate)) / (1024.0 * 1024.0 * 1024.0)});

    // Validate parameter count
    const param_count = try cfg.model.countParameters();
    std.log.info("Total parameters: {d}", .{param_count});

    const target_params: u64 = 1_000_000_000_000; // 1T
    const tolerance: f64 = 0.05; // 5% tolerance
    const ratio = @as(f64, @floatFromInt(param_count)) / @as(f64, @floatFromInt(target_params));

    if (ratio < (1.0 - tolerance) or ratio > (1.0 + tolerance)) {
        std.log.warn("Parameter count differs from 1T target by more than 5%", .{});
    }

    std.log.info("Configuration valid!", .{});
}

fn runProfile(allocator: std.mem.Allocator, args: *std.process.ArgIterator) !void {
    _ = allocator;
    _ = args;

    std.log.err("Profiling requires Nsight Systems or Compute to be installed.", .{});
    std.log.info("Use: nsys profile efla-train train --config ...", .{});
    std.log.info("Or:  ncu efla-train train --config ...", .{});
}
