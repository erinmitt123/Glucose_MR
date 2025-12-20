using System;
using System.Collections.Generic;
using UnityEngine;

namespace WordsToNumbers
{
    internal static class Compiler
    {
        /// <summary>
        /// Converts a text and its numeric regions into either a number (if single region) 
        /// or replaces numeric regions in the text with their numeric values.
        /// </summary>
        public static string Compile(string text, List<Region> regions)
        {
            if (regions == null || regions.Count == 0)
                return text;

            // Single region spans entire text: return just the number
            if (regions.Count == 1 && regions[0].Start == 0 && regions[0].End == text.Length - 1)
                return GetNumberWithDecimalPlaces(regions[0]);

            // Otherwise, replace numeric regions in text
            return ReplaceRegionsInText(text, regions);
        }

        private static string GetNumberWithDecimalPlaces(Region region)
        {
            var number = GetNumber(region);
            return number.decimalPlaces > 0
                ? number.val.ToString($"F{number.decimalPlaces}")
                : number.val.ToString();
        }


        /// <summary>
        /// Computes the numeric value of a Region by processing its SubRegions.
        /// </summary>
        private static (double val, int decimalPlaces) GetNumber(Region region)
        {
            double sum = 0.0;
            double scale = 1.0;

            bool decimalReached = false;
            var decimals = new List<SubRegion>();
            int decimalPlaces = 0;

            foreach (var sub in region.SubRegions)
            {
                if (sub.Type == TokenType.Decimal)
                {
                    decimalReached = true;
                    continue;
                }

                if (decimalReached)
                {
                    decimals.Add(sub);
                    continue;
                }

                sum += ComputeSubRegionValue(sub);
            }

            foreach (var sub in decimals)
            {
                double value = 0;
                foreach (var token in sub.Tokens)
                {
                    value += Constants.NUMBER[token.Lower];
                    Debug.Log($"DECIMALS: The token is {token.Lower}");
                }
                Debug.Log($"DECIMALS: The final val is {value}");
                int digits = value < 10 ? 1 : (int)Math.Floor(Math.Log10(value) + 1);
                for (int i = 0; i < digits; i++) scale /= 10.0;

                sum += value * scale;
                decimalPlaces += digits;
            }

            return (sum, decimalPlaces);
        }

        /// <summary>
        /// Computes the numeric value of a SubRegion
        ///     - Handles HUNDRED and MAGNITUDE multipliers.
        ///     - Skips tokens as needed
        ///     - Preserves order and combination logic
        /// </summary>
        private static double ComputeSubRegionValue(SubRegion sub)
        {
            var tokens = sub.Tokens;
            int count = tokens.Count;
            List<double> processedTokens = new();

            for (int i = 0; i < count; i++)
            {
                var token = tokens[i];
                if (!Constants.NUMBER.TryGetValue(token.Lower, out var value))
                    continue;

                // Handle Hundred and Magnitude tokens
                if (token.Type == TokenType.Hundred || token.Type == TokenType.Magnitude)
                {
                    double multiplier = token.Type == TokenType.Hundred ? 100 : 1;
                    double afterSum = 0;

                    // Compute tokens to add after this one, like a filter
                    for (int j = i + 1; j < count; j++)
                    {
                        var nextToken = tokens[j];
                        if (!Constants.NUMBER.TryGetValue(nextToken.Lower, out var nextValue))
                            continue;

                        // Skip tokens that are HUNDRED or TEN after a HUNDRED
                        if (nextToken.Type == TokenType.Hundred) break;
                        if (j > 0 && tokens[j - 1].Type == TokenType.Hundred && nextToken.Type == TokenType.Ten) break;
                        Debug.Log($"The next value is {nextValue}");
                        afterSum += nextValue;
                    }
                    Debug.Log($"The calculated value in the loop is {value * multiplier + afterSum}");
                    processedTokens.Add(value * multiplier + afterSum);
                }
                else
                {
                    if (i > 0 && tokens[i - 1].Type == TokenType.Hundred) continue;
                    if (i > 1 && tokens[i - 2].Type == TokenType.Hundred && tokens[i - 1].Type == TokenType.Ten) continue;

                    processedTokens.Add(value);
                    Debug.Log($"The straight up value is {value}");
                }
            }

            double subSum = 1;
            foreach (double value in processedTokens) { 
            Debug.Log($"Value being multiplied into is {value}");
            subSum *= value;
            }
            return subSum;
        }

        /// <summary>
        /// Replaces numeric regions in text with their numeric values.
        /// </summary>
        private static string ReplaceRegionsInText(string text, List<Region> regions)
        {
            var result = text;
            int offset = 0;

            foreach (var region in regions)
            {
                var replacement = GetNumberWithDecimalPlaces(region);
                int length = region.End - region.Start + 1;
                result = StringUtils.Splice(result, region.Start + offset, length, replacement);
                offset += replacement.Length - length;
            }

            return result;
        }

    }
}