// Copyright 2026 The dotnet-embedding-parity Authors
// Licensed under the Apache License, Version 2.0

using System.Globalization;
using System.Reflection;
using System.Text;
using System.Text.Json;
using Microsoft.ML.Tokenizers;

// Usage: OracleMlTokenizers <vocab.txt>
// Prints BertTokenizer output for the inputs that diverged, under default BertOptions and
// under single-option changes, so each divergence is pinned to a package default or a
// package defect independently of ElBruno.LocalEmbeddings.
// Measurement: OracleMlTokenizers ids <vocab.txt> <probes.json> <out.json>
// Writes the token ids for every probe under both configurations below, with the
// globalization mode the process actually ran in, for reference/score_ids.py to check
// against Hugging Face. Run it once as built and once with -p:InvariantGlobalization=true.
if (args.Length == 4 && args[0] == "ids")
{
    var invariant = AppContext.TryGetSwitch("System.Globalization.Invariant", out var on) && on;
    var ids = new Dictionary<string, Dictionary<string, int[]>>();
    using (var doc = JsonDocument.Parse(File.ReadAllText(args[2])))
    {
        foreach (var (name, options) in Configurations())
        {
            var tokenizer = BertTokenizer.Create(args[1], options);
            var perProbe = new Dictionary<string, int[]>();
            foreach (var probe in doc.RootElement.GetProperty("probes").EnumerateArray())
                perProbe[probe.GetProperty("id").GetString()!] = [.. tokenizer.EncodeToIds(probe.GetProperty("text").GetString()!)];
            ids[name] = perProbe;
        }
    }
    var result = new Dictionary<string, object>
    {
        ["Package"] = "Microsoft.ML.Tokenizers",
        ["PackageVersion"] = typeof(BertTokenizer).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion ?? "unknown",
        ["Runtime"] = System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription,
        ["InvariantGlobalization"] = invariant,
        ["NormalizeFormDLengthOfE9"] = "\u00e9".Normalize(NormalizationForm.FormD).Length,
        ["Ids"] = ids,
    };
    Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(args[3])) ?? ".");
    File.WriteAllText(args[3], JsonSerializer.Serialize(result));
    Console.WriteLine($"invariant={invariant}: ids for {ids.First().Value.Count} probes x {ids.Count} configurations -> {args[3]}");
    return 0;
}

if (args.Length != 1)
{
    Console.Error.WriteLine("usage: OracleMlTokenizers <vocab.txt> | ids <vocab.txt> <probes.json> <out.json>");
    return 2;
}

string[] inputs =
[
    "Café crème brûlée in São Paulo, naïve résumé",
    "one\nline",
    "one\tline",
    "one \nline",
    "  line one\n\tline two  ",
    "東京タワーの夜景 🚀 is beautiful",
    "🚀",
];

var vocab = File.ReadAllLines(args[0]);

foreach (var (name, options) in Configurations())
{
    Console.WriteLine($"== {name}: LowerCase={options.LowerCaseBeforeTokenization}, " +
        $"BasicTokenization={options.ApplyBasicTokenization}, CJK={options.IndividuallyTokenizeCjk}, " +
        $"RemoveNonSpacingMarks={options.RemoveNonSpacingMarks}");
    var tokenizer = BertTokenizer.Create(args[0], options);
    foreach (var input in inputs)
    {
        var ids = tokenizer.EncodeToIds(input);
        var tokens = new List<string>(ids.Count);
        foreach (var id in ids)
            tokens.Add(id >= 0 && id < vocab.Length ? vocab[id] : "?");
        Console.WriteLine($"  {Escape(input),-48} -> {string.Join(' ', tokens)}");
    }
}
return 0;

static (string Name, BertOptions Options)[] Configurations() =>
[
    ("default BertOptions", new BertOptions()),
    ("RemoveNonSpacingMarks = true", new BertOptions { RemoveNonSpacingMarks = true }),
];

static string Escape(string s) => s.Replace("\n", "\\n").Replace("\t", "\\t");
