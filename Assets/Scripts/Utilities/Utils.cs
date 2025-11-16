using UnityEngine;

public static class Utils
{
    public static double? GetDoubleFromNonnumericalSentence(string sentence)
    {
        if (sentence == null) { return null; }

        string newSentence = WordNumberParser.ConvertWordNumbersInSentence(sentence);
        var number = DoubleFromStringParser.ParseDoubleFromString(newSentence);

        return number;
    }
}
