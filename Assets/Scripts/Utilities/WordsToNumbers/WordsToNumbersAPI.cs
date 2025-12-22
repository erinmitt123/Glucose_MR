using System.Collections.Generic;
using UnityEngine;

namespace WordsToNumbers
{
    public static class WordsToNumbersAPI
    {
        public static string Convert(string text, bool fuzzy = false, bool impliedHundreds = false, bool canDigitByDigit = true)
        {
            List<Region> regions = Parser.Parse(text, fuzzy, impliedHundreds);
            if (regions.Count == 0) return text;

            foreach (Region region in regions)
            {
                for (int i = 0; i < region.SubRegions.Count; i++) 
                {
                    var subRegion = region.SubRegions[i];
                    foreach (Token token in subRegion.Tokens)
                    {
                        Debug.Log($"In Region Start: {region.Start}, SubRegion: {i}, we are at {token.Value} ");
                    }
                }
            }

            string compiled = Compiler.Compile(text, regions, canDigitByDigit);
            return compiled;
        }
    }
}