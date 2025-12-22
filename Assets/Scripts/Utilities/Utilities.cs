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

internal static class NumUtils
{
    /// <summary>
    /// A simple helper to multiply powers of 10 using a while loop.
    /// Eliminates costly overhead of Math.Pow for small, int exponents
    /// </summary>
    /// <param name="exp">The exponent or power of 10 needed</param>
    /// <returns> Ten raised to the power of <paramref name="exp"/></returns>
    public static int Pow10(int exp)
    {
        int result = 1;
        while (exp-- > 0) result *= 10;
        return result;
    }

    /// <summary>
    /// Counts the number of digits present in a given integer.
    /// </summary>
    /// <param name="n">The number to count</param>
    /// <returns>The number of digits present in <paramref name="n"/></returns>
    public static int DigitCount(int n)
    {
        if (n == 0) return 1;
        int count = 0;
        while (n > 0)
        {
            n /= 10;
            count++;
        }
        return count;
    }
}
