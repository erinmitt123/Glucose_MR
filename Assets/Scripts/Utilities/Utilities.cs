using System;
using WordsToNumbers;

internal static class Utils
{
    public static double? GetDoubleFromNonnumericalSentence(string text)
    {
        if (text == null) { return null; }

        string newText = WordsToNumbersAPI.Convert(text);
        var number = DoubleFromStringParser.ParseDoubleFromString(newText);

        return number;
    }
}

internal static class StringUtils
{
    public static string Splice(string str, int index, int count, string add)
    {
        if (index < 0) index = Math.Max(0, str.Length + index);
        return str.Substring(0, index) + add + str.Substring(index + count);
    }
}
