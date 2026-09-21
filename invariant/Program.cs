// Copyright 2026 The dotnet-embedding-parity Authors
// Licensed under the Apache License, Version 2.0

using System.Text;

// Usage: dotnet run --project invariant [-p:InvariantGlobalization=false]
// Uncased BERT strips accents by decomposing text (FormD) and dropping the combining marks.
// Under invariant globalization Normalize returns its input unchanged, by design, and
// does not throw: "é" stays one code unit, there is no mark to drop, and the accent survives.
var decomposed = "\u00e9".Normalize(NormalizationForm.FormD);
Console.WriteLine($"\"\\u00e9\".Normalize(FormD) -> {decomposed.Length} code unit(s): " +
    string.Join(' ', decomposed.Select(c => $"U+{(int)c:X4}")));

// The guard: check once at startup, and refuse rather than embed wrongly in silence.
if (decomposed.Length != 2)
{
    Console.Error.WriteLine("Refusing to start: Unicode decomposition is unavailable (invariant globalization), " +
        "so accent stripping would silently do nothing. Set <InvariantGlobalization>false</InvariantGlobalization>, " +
        "or run on an image that carries ICU, such as a chiseled -extra variant.");
    return 1;
}
Console.WriteLine("Decomposition available: accent stripping will work.");
return 0;
