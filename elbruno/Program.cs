// Copyright 2026 The dotnet-embedding-parity Authors
// Licensed under the Apache License, Version 2.0

using ElBruno.LocalEmbeddings;
using ElBruno.LocalEmbeddings.Options;
using Microsoft.Extensions.AI;
using OracleCompetitors;

// Usage: OracleElBruno <texts.json> <cache-dir> <out.json>
// Default LocalEmbeddingsOptions, as in the README quick start: the library downloads
// sentence-transformers/all-MiniLM-L6-v2 itself. Only the cache directory is redirected,
// so the file it loaded can be hashed.
// Diagnostic: OracleElBruno tokens <texts.json> <model-dir> <out.json>
// Dumps the token ids produced by the library's own public Tokenizer, the class
// LocalEmbeddingGenerator uses, so a divergence can be attributed to tokenization.
if (args.Length == 4 && args[0] == "tokens")
{
    var tokenizer = new Tokenizer(args[2], maxLength: 512);
    var dump = new Dictionary<string, long[]>();
    foreach (var t in ProbeOutput.ReadTexts(args[1]))
    {
        var (ids, mask) = tokenizer.Tokenize(t.Text);
        dump[t.Id] = ProbeOutput.Unpadded(ids, mask);
    }
    ProbeOutput.WriteTokens(args[3], dump);
    return 0;
}

if (args.Length != 3)
{
    Console.Error.WriteLine("usage: OracleElBruno <texts.json> <cache-dir> <out.json>");
    return 2;
}

var texts = ProbeOutput.ReadTexts(args[0]);
var options = new LocalEmbeddingsOptions { CacheDirectory = Path.GetFullPath(args[1]) };

await using var generator = await LocalEmbeddingGenerator.CreateAsync(options, progress: null);

var (single, batch) = await ProbeOutput.RunAsync(
    texts,
    async text => (await generator.GenerateAsync([text]))[0].Vector.ToArray(),
    async many =>
    {
        var generated = await generator.GenerateAsync(many);
        var vectors = new float[generated.Count][];
        for (int i = 0; i < generated.Count; i++)
            vectors[i] = generated[i].Vector.ToArray();
        return vectors;
    });

string? modelFile = null;
foreach (var f in Directory.EnumerateFiles(options.CacheDirectory, "*.onnx", SearchOption.AllDirectories))
{
    modelFile = modelFile is null ? f : modelFile + ";" + f;
}

ProbeOutput.Write(args[2], new ProbeOutput.Result(
    Library: "ElBruno.LocalEmbeddings",
    Package: "ElBruno.LocalEmbeddings",
    PackageVersion: ProbeOutput.VersionOf(typeof(LocalEmbeddingGenerator).Assembly),
    OnnxRuntimeVersion: ProbeOutput.OnnxRuntimeVersion(),
    ModelFile: modelFile,
    ModelSha256: modelFile is not null && !modelFile.Contains(';') ? ProbeOutput.Sha256(modelFile) : null,
    Configuration: new Dictionary<string, string>
    {
        ["ModelName"] = options.ModelName,
        ["MaxSequenceLength"] = options.MaxSequenceLength.ToString(),
        ["NormalizeEmbeddings"] = options.NormalizeEmbeddings.ToString(),
        ["PreferQuantized"] = options.PreferQuantized.ToString(),
    },
    Single: single,
    Batch: batch));
return 0;
