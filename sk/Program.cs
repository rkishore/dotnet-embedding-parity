// Copyright 2026 The dotnet-embedding-parity Authors
// Licensed under the Apache License, Version 2.0

using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.SemanticKernel.Connectors.Onnx;
using OracleCompetitors;

// Usage: OracleSk <texts.json> <model.onnx> <vocab.txt> <out.json>
// Default BertOnnxOptions: what a user of AddBertOnnxEmbeddingGenerator gets.
if (args.Length != 4)
{
    Console.Error.WriteLine("usage: OracleSk <texts.json> <model.onnx> <vocab.txt> <out.json>");
    return 2;
}

var texts = ProbeOutput.ReadTexts(args[0]);
var options = new BertOnnxOptions();

await using var provider = new ServiceCollection()
    .AddBertOnnxEmbeddingGenerator(args[1], args[2], options)
    .BuildServiceProvider();
var generator = provider.GetRequiredService<IEmbeddingGenerator<string, Embedding<float>>>();

var (single, batch) = await ProbeOutput.RunAsync(
    texts,
    async text => (await generator.GenerateAsync(text)).Vector.ToArray(),
    async many =>
    {
        var generated = await generator.GenerateAsync(many);
        var vectors = new float[generated.Count][];
        for (int i = 0; i < generated.Count; i++)
            vectors[i] = generated[i].Vector.ToArray();
        return vectors;
    });

ProbeOutput.Write(args[3], new ProbeOutput.Result(
    Library: "Semantic Kernel Connectors.Onnx",
    Package: "Microsoft.SemanticKernel.Connectors.Onnx",
    PackageVersion: ProbeOutput.VersionOf(typeof(BertOnnxOptions).Assembly),
    OnnxRuntimeVersion: ProbeOutput.OnnxRuntimeVersion(),
    ModelFile: Path.GetFullPath(args[1]),
    ModelSha256: ProbeOutput.Sha256(args[1]),
    Configuration: new Dictionary<string, string>
    {
        ["PoolingMode"] = options.PoolingMode.ToString(),
        ["NormalizeEmbeddings"] = options.NormalizeEmbeddings.ToString(),
        ["MaximumTokens"] = options.MaximumTokens.ToString(),
        ["CaseSensitive"] = options.CaseSensitive.ToString(),
        ["UnicodeNormalization"] = options.UnicodeNormalization.ToString(),
    },
    Single: single,
    Batch: batch));
return 0;
