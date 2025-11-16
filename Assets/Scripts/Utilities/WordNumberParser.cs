using System;
using System.Collections.Generic;
using System.Text;
using System.Text.RegularExpressions;

public static class WordNumberParser
{
    public static readonly Dictionary<string, int> Units = new Dictionary<string, int>
    {
        {"zero",0},{"one",1},{"two",2},{"three",3},{"four",4},{"five",5},
        {"six",6},{"seven",7},{"eight",8},{"nine",9},{"ten",10},
        {"eleven",11},{"twelve",12},{"thirteen",13},{"fourteen",14},
        {"fifteen",15},{"sixteen",16},{"seventeen",17},{"eighteen",18},{"nineteen",19}
    };

    public static readonly Dictionary<string, int> Tens = new Dictionary<string, int>
    {
        {"twenty",20},{"thirty",30},{"forty",40},{"fifty",50},
        {"sixty",60},{"seventy",70},{"eighty",80},{"ninety",90}
    };

    private static string CleanToken(string token)
    {
        return Regex.Replace(token, @"^[^A-Za-z0-9\-]+|[^A-Za-z0-9\-]+$", "")
                     .ToLowerInvariant();
    }

    public static string ConvertWordNumbersInSentence(string input)
    {
        if (string.IsNullOrEmpty(input)) return input;

        string[] originalTokens = input.Split(' ');
        string[] cleanTokens = new string[originalTokens.Length];

        for (int i = 0; i < originalTokens.Length; i++)
            cleanTokens[i] = CleanToken(originalTokens[i]);

        var output = new StringBuilder();
        int index = 0;

        while (index < originalTokens.Length)
        {
            int lookahead = index;

            if (TryParseFullNumber(cleanTokens, ref lookahead, out string numericStr))
            {
                output.Append(numericStr);

                // Preserve punctuation from last original token
                string lastOriginal = originalTokens[lookahead - 1];
                Match m = Regex.Match(lastOriginal, @"([^\w\-]+)$");
                if (m.Success) output.Append(m.Value);

                index = lookahead;
            }
            else
            {
                output.Append(originalTokens[index]);
                index++;
            }

            if (index < originalTokens.Length)
                output.Append(" ");
        }

        return output.ToString().Trim();
    }

    private static bool TryParseFullNumber(string[] tokens, ref int index, out string result)
    {
        result = null;
        int startIndex = index;

        if (!TryParseWhole(tokens, ref index, out int whole))
        {
            index = startIndex;
            return false;
        }

        // Decimal part
        if (index < tokens.Length && tokens[index] == "point")
        {
            index++; // skip "point"
            var decimalSb = new StringBuilder();

            while (index < tokens.Length)
            {
                string t = tokens[index];

                if (Units.TryGetValue(t, out int unit))
                {
                    decimalSb.Append(unit); // single digit
                    index++;
                }
                else if (Tens.TryGetValue(t, out int ten))
                {
                    decimalSb.Append(ten); // two digits
                    index++;
                    // Check if a following unit exists, e.g., "twenty one"
                    if (index < tokens.Length && Units.TryGetValue(tokens[index], out int extra))
                    {
                        decimalSb.Append(extra);
                        index++;
                    }
                }
                else
                {
                    break;
                }
            }

            result = decimalSb.Length > 0 ? $"{whole}.{decimalSb}" : whole.ToString();
        }
        else
        {
            result = whole.ToString();
        }

        return true;
    }

    private static bool TryParseWhole(string[] tokens, ref int index, out int value)
    {
        value = 0;
        int startIndex = index;
        bool found = false;
        int acc = 0;

        while (index < tokens.Length)
        {
            string t = tokens[index];

            if (string.IsNullOrEmpty(t)) break;
            if (t == "and") { index++; continue; }

            if (Units.TryGetValue(t, out int u))
            {
                acc += u;
                index++;
                found = true;
                continue;
            }

            if (Tens.TryGetValue(t, out int ten))
            {
                acc += ten;
                index++;
                if (index < tokens.Length && Units.TryGetValue(tokens[index], out int extra))
                {
                    acc += extra;
                    index++;
                }
                found = true;
                continue;
            }

            if (t == "hundred")
            {
                acc = acc == 0 ? 100 : acc * 100;
                index++;
                found = true;
                continue;
            }

            break;
        }

        if (!found)
        {
            index = startIndex;
            return false;
        }

        value = acc;
        return true;
    }
}
