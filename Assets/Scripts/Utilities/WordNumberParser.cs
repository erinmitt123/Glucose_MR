using System;
using System.Collections.Generic;
using System.Text;
using System.Text.RegularExpressions;

public static class WordNumberParser
{
    private static readonly Dictionary<string, int> NumberWords = new(StringComparer.OrdinalIgnoreCase)
    {
        // Basic numbers
        { "zero", 0 }, { "oh", 0 }, { "one", 1 }, { "two", 2 }, { "three", 3 },
        { "four", 4 }, { "five", 5 }, { "six", 6 }, { "seven", 7 }, { "eight", 8 }, { "nine", 9 },

        // Teens
        { "ten", 10 }, { "eleven", 11 }, { "twelve", 12 }, { "thirteen", 13 }, { "fourteen", 14 },
        { "fifteen", 15 }, { "sixteen", 16 }, { "seventeen", 17 }, { "eighteen", 18 }, { "nineteen", 19 },

        // Tens
        { "twenty", 20 }, { "thirty", 30 }, { "forty", 40 }, { "fifty", 50 },
        { "sixty", 60 }, { "seventy", 70 }, { "eighty", 80 }, { "ninety", 90 }
    };

    private static readonly HashSet<string> Multipliers = new(StringComparer.OrdinalIgnoreCase)
    {
        "hundred", "thousand", "million"
    };

    public static string ConvertWordNumbersInSentence(string sentence)
    {
        if (string.IsNullOrWhiteSpace(sentence))
            return sentence;

        string expanded = ExpandHyphens(sentence).ToLower();
        string[] tokens = expanded.Split(' ', StringSplitOptions.RemoveEmptyEntries);

        double result = 0;
        double current = 0;
        bool inDecimal = false;
        double decimalPlace = 1.0;

        int? spanStart = null;
        int spanEnd = -1;

        for (int i = 0; i < tokens.Length; i++)
        {
            string rawWord = tokens[i];
            string word = rawWord.Trim();

            bool isNumberWord =
                NumberWords.ContainsKey(word) ||
                word == "point" ||
                Multipliers.Contains(word);

            // Mark beginning of number-span
            if (isNumberWord && spanStart == null)
                spanStart = i;

            if (word == "and")
                continue;

            if (word == "point")
            {
                inDecimal = true;
                spanEnd = i;
                continue;
            }

            if (inDecimal)
            {
                if (NumberWords.TryGetValue(word, out int number))
                {
                    int digitCount = DigitCount(number);
                    decimalPlace *= Math.Pow(0.1, digitCount);

                    current += number * decimalPlace;
                    spanEnd = i;
                }
                continue;
            }

            if (NumberWords.TryGetValue(word, out int val))
            {
                current += val;
                spanEnd = i;
            }
            else if (word == "hundred")
            {
                current *= 100;
                spanEnd = i;
            }
            else if (word == "thousand")
            {
                current *= 1000;
                result += current;
                current = 0;
                spanEnd = i;
            }
            else if (word == "million")
            {
                current *= 1_000_000;
                result += current;
                current = 0;
                spanEnd = i;
            }
        }

        result += current;

        if (spanStart == null)
            return sentence; // no number found

        return ReplaceFirstNumber(sentence, result, spanStart.Value, spanEnd);
    }

    // NEW: Replace the correct region instead of appending at the end
    private static string ReplaceFirstNumber(string originalSentence, double number, int startToken, int endToken)
    {
        // Work on original spacing/punctuation
        string[] origTokens = originalSentence.Split(' ');

        // Safety check if token counts mismatch
        if (startToken >= origTokens.Length)
            return originalSentence + $" {number}";

        var sb = new StringBuilder();

        // Add text BEFORE the number region
        for (int i = 0; i < startToken; i++)
        {
            sb.Append(origTokens[i]);
            if (i < origTokens.Length - 1) sb.Append(" ");
        }

        // Determine if the last token has trailing punctuation
        string lastToken = origTokens[endToken];
        string punctuation = Regex.Match(lastToken, @"[^\w\-]+$").Value;

        // Add the number
        sb.Append(number.ToString());
        sb.Append(punctuation);

        // Add text AFTER the number region
        for (int i = endToken + 1; i < origTokens.Length; i++)
        {
            sb.Append(" ");
            sb.Append(origTokens[i]);
        }

        return sb.ToString();
    }

    private static string ExpandHyphens(string sentence)
    {
        StringBuilder sb = new(sentence.Length);
        foreach (char c in sentence)
            sb.Append(c == '-' ? ' ' : c);
        return sb.ToString();
    }

    private static int DigitCount(int number)
    {
        number = Math.Abs(number);
        if (number == 0) return 1;
        return (int)Math.Floor(Math.Log10(number)) + 1;
    }
}