// Copyright 2026 The dotnet-embedding-parity Authors
// Licensed under the Apache License, Version 2.0

using System.Reflection;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace OracleCompetitors;

/// <summary>
/// Shared plumbing for the per-library probes: reads the probe texts, runs a library's
/// single-text and multi-text embedding calls, and writes one JSON result that
/// <c>compare.py</c> scores against the reference. Deliberately free of any library
/// reference, so each probe project restores only its own ONNX Runtime.
/// </summary>
public static class ProbeOutput
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = false,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };

    public sealed record ProbeText(string Id, string Text);

    public sealed record SingleResult(string Id, float[]? Vector, string? Error);

    public sealed record BatchResult(string[] Ids, float[][]? Vectors, string? Error);

    public sealed record Result(
        string Library,
        string Package,
        string? PackageVersion,
        string? OnnxRuntimeVersion,
        string? ModelFile,
        string? ModelSha256,
        IReadOnlyDictionary<string, string> Configuration,
        IReadOnlyList<SingleResult> Single,
        BatchResult Batch);

    public static IReadOnlyList<ProbeText> ReadTexts(string path)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        var list = new List<ProbeText>();
        foreach (var t in doc.RootElement.GetProperty("texts").EnumerateArray())
        {
            var id = t.GetProperty("id").GetString() ?? throw new InvalidDataException("text without id");
            var text = t.GetProperty("text").GetString() ?? throw new InvalidDataException($"{id} has no text");
            list.Add(new ProbeText(id, text));
        }
        return list;
    }

    /// <summary>Embeds each text on its own, then every non-empty text in one call.</summary>
    public static async Task<(List<SingleResult> Single, BatchResult Batch)> RunAsync(
        IReadOnlyList<ProbeText> texts,
        Func<string, Task<float[]>> embedOne,
        Func<IReadOnlyList<string>, Task<float[][]>> embedMany)
    {
        var single = new List<SingleResult>(texts.Count);
        foreach (var t in texts)
        {
            try
            {
                single.Add(new SingleResult(t.Id, await embedOne(t.Text), null));
            }
            catch (Exception ex)
            {
                single.Add(new SingleResult(t.Id, null, $"{ex.GetType().Name}: {ex.Message}"));
            }
        }

        // The empty string is excluded from the batch so that one edge-case failure
        // cannot void the whole multi-text call.
        var ids = new List<string>();
        var batchTexts = new List<string>();
        foreach (var t in texts)
        {
            if (t.Text.Length == 0)
                continue;
            ids.Add(t.Id);
            batchTexts.Add(t.Text);
        }

        BatchResult batch;
        try
        {
            batch = new BatchResult(ids.ToArray(), await embedMany(batchTexts), null);
        }
        catch (Exception ex)
        {
            batch = new BatchResult(ids.ToArray(), null, $"{ex.GetType().Name}: {ex.Message}");
        }

        return (single, batch);
    }

    public static string? VersionOf(Assembly assembly) =>
        assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion;

    public static string? OnnxRuntimeVersion()
    {
        foreach (var a in AppDomain.CurrentDomain.GetAssemblies())
        {
            if (a.GetName().Name == "Microsoft.ML.OnnxRuntime")
                return VersionOf(a) ?? a.GetName().Version?.ToString();
        }
        return null;
    }

    public static string? Sha256(string? path)
    {
        if (path is null || !File.Exists(path))
            return null;
        using var stream = File.OpenRead(path);
        return Convert.ToHexStringLower(SHA256.HashData(stream));
    }

    /// <summary>Keeps the ids whose attention mask is 1; drops padding.</summary>
    public static long[] Unpadded(long[] ids, long[] mask)
    {
        var kept = new List<long>(ids.Length);
        for (int i = 0; i < ids.Length; i++)
        {
            if (mask[i] != 0)
                kept.Add(ids[i]);
        }
        return kept.ToArray();
    }

    public static void WriteTokens(string path, IReadOnlyDictionary<string, long[]> tokens)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path)) ?? ".");
        File.WriteAllText(path, JsonSerializer.Serialize(tokens, JsonOptions));
        Console.WriteLine($"wrote token ids for {tokens.Count} texts to {path}");
    }

    /// <summary>
    /// Every runner serialises through here, so this is the one place that keeps absolute
    /// paths out of committed results: <c>ModelFile</c> is rewritten relative to the
    /// repository root, or reduced to its file name when it lies outside the repository.
    /// </summary>
    public static void Write(string path, Result result)
    {
        result = result with { ModelFile = RepositoryRelative(result.ModelFile) };
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path)) ?? ".");
        File.WriteAllText(path, JsonSerializer.Serialize(result, JsonOptions));
        var failures = 0;
        foreach (var s in result.Single)
        {
            if (s.Error is not null)
                failures++;
        }
        Console.WriteLine(
            $"{result.Library}: {result.Single.Count - failures}/{result.Single.Count} single ok, " +
            $"batch {(result.Batch.Error is null ? "ok" : "failed: " + result.Batch.Error)}, model {result.ModelFile}");
    }

    /// <summary>ElBruno's <c>ModelFile</c> may join several paths with ';'; each is rewritten.</summary>
    private static string? RepositoryRelative(string? modelFile)
    {
        if (modelFile is null)
            return null;
        var root = RepositoryRoot();
        var parts = modelFile.Split(';');
        for (int i = 0; i < parts.Length; i++)
        {
            var full = Path.GetFullPath(parts[i]);
            var relative = root is null ? null : Path.GetRelativePath(root, full);
            parts[i] = relative is null || relative.StartsWith("..", StringComparison.Ordinal) || Path.IsPathRooted(relative)
                ? "<outside repository>/" + Path.GetFileName(full)
                : relative.Replace(Path.DirectorySeparatorChar, '/');
        }
        return string.Join(';', parts);
    }

    /// <summary>The nearest ancestor of the build output that holds Directory.Build.props.</summary>
    private static string? RepositoryRoot()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            if (File.Exists(Path.Combine(dir.FullName, "Directory.Build.props")))
                return dir.FullName;
        }
        return null;
    }
}
