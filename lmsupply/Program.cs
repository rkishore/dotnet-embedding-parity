// Copyright 2026 The dotnet-embedding-parity Authors
// Licensed under the Apache License, Version 2.0

using LMSupply;
using LMSupply.Embedder;
using LMSupply.Text;
using OracleCompetitors;

// Usage: OracleLmSupply <texts.json> <cache-dir> <out.json>
// Default EmbedderOptions with the Hugging Face repo id, as in the README: the library
// auto-discovers and downloads the ONNX file itself. Only the cache directory is
// redirected. The file it chose is recorded from IModelRuntimeInfo.ModelPath.
// Diagnostic: OracleLmSupply tokens <texts.json> <tokenizer-dir> <out.json>
// Dumps the token ids from TokenizerFactory.CreateAutoSequenceAsync, the call
// LocalEmbedder.LoadAsync makes (LocalEmbedder.cs:205), with its default 512 cap.
if (args.Length == 4 && args[0] == "tokens")
{
    var tokenizer = await TokenizerFactory.CreateAutoSequenceAsync(args[2], 512);
    var dump = new Dictionary<string, long[]>();
    foreach (var t in ProbeOutput.ReadTexts(args[1]))
    {
        var encoded = tokenizer.EncodeSequence(t.Text);
        dump[t.Id] = ProbeOutput.Unpadded(encoded.InputIds, encoded.AttentionMask);
    }
    ProbeOutput.WriteTokens(args[3], dump);
    return 0;
}

if (args.Length != 3)
{
    Console.Error.WriteLine("usage: OracleLmSupply <texts.json> <cache-dir> <out.json>");
    return 2;
}

var texts = ProbeOutput.ReadTexts(args[0]);
var options = new EmbedderOptions { CacheDirectory = Path.GetFullPath(args[1]) };

await using var model = await LocalEmbedder.LoadAsync("sentence-transformers/all-MiniLM-L6-v2", options);

var (single, batch) = await ProbeOutput.RunAsync(
    texts,
    async text => await model.EmbedAsync(text),
    async many => await model.EmbedAsync(many));

var runtime = model as IModelRuntimeInfo;

ProbeOutput.Write(args[2], new ProbeOutput.Result(
    Library: "LMSupply.Embedder",
    Package: "LMSupply.Embedder",
    PackageVersion: ProbeOutput.VersionOf(typeof(LocalEmbedder).Assembly),
    OnnxRuntimeVersion: ProbeOutput.OnnxRuntimeVersion(),
    ModelFile: runtime?.ModelPath,
    ModelSha256: ProbeOutput.Sha256(runtime?.ModelPath),
    Configuration: new Dictionary<string, string>
    {
        ["MaxSequenceLength"] = options.MaxSequenceLength.ToString(),
        ["NormalizeEmbeddings"] = options.NormalizeEmbeddings.ToString(),
        ["PoolingMode"] = options.PoolingMode.ToString(),
        ["DoLowerCase"] = options.DoLowerCase.ToString(),
        ["Provider"] = options.Provider.ToString(),
        ["ActiveProviders"] = runtime is null ? "unknown" : string.Join(",", runtime.ActiveProviders),
    },
    Single: single,
    Batch: batch));
return 0;
