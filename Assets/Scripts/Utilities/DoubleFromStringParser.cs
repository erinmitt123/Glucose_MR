using System.Globalization;
using System.Text.RegularExpressions;

public static class DoubleFromStringParser
{
    private static readonly string regexPattern = @"\d+(\.\d+)?";

    public static double? ParseDoubleFromString(string sentence)
    {
        if (string.IsNullOrEmpty(sentence))
            return null;

        Match match = Regex.Match(sentence, regexPattern);

        if (!match.Success)
        {
            return null;
        }

        string floatString = match.Value;

        if (double.TryParse(floatString, NumberStyles.Float, CultureInfo.InvariantCulture, out double number))
            return number;

        return null; // fallback for extremely rare parse failures
    }
}
