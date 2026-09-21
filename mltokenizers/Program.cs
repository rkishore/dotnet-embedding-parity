// Copyright 2026 The dotnet-embedding-parity Authors
// Licensed under the Apache License, Version 2.0

using Microsoft.ML.Tokenizers;

// Usage: OracleMlTokenizers <vocab.txt>
// Prints BertTokenizer output for the inputs that diverged, under default BertOptions and
// under single-option changes, so each divergence is pinned to a package default or a
// package defect independently of ElBruno.LocalEmbeddings.
if (args.Length != 1)
{
    Console.Error.WriteLine("usage: OracleMlTokenizers <vocab.txt>");
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

var configurations = new (string Name, BertOptions Options)[]
{
    ("default BertOptions", new BertOptions()),
    ("RemoveNonSpacingMarks = true", new BertOptions { RemoveNonSpacingMarks = true }),
};

var vocab = File.ReadAllLines(args[0]);

foreach (var (name, options) in configurations)
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

static string Escape(string s) => s.Replace("\n", "\\n").Replace("\t", "\\t");
