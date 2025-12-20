using System.Collections.Generic;
using UnityEngine;

namespace WordsToNumbers
{
    public static class WordsToNumbersAPI
    {
        public static string Convert(string text, bool fuzzy = false, bool impliedHundreds = false)
        {
            List<Region> regions = Parser.Parse(text, fuzzy, impliedHundreds);
            if (regions.Count == 0) return text;

            foreach (Region region in regions)
            {
                foreach (Token token in region.Tokens)
                {
                    Debug.Log($"In Region Start: {region.Start}, we are at {token.Value} ");
                }
            }

            string compiled = Compiler.Compile(text, regions);
            return compiled;
        }
    }
}