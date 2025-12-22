using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using UnityEngine;
using UnityEngine.Rendering;

namespace WordsToNumbers
{
    internal static class Compiler
    {
        /// <summary>
        /// Converts a text and its numeric regions into either a number (if single region)
        /// or replaces numeric regions in the text with their numeric values.
        /// </summary>
        public static string Compile(string text, List<Region> regions, bool canDigitByDigit)
        {
            if (regions == null || regions.Count == 0)
                return text;

            // Combine adjacent regions if permission granted
            if (canDigitByDigit)
                regions = CombineAdjacentRegions(text, regions);

            // Single region spans entire text: return just the number
            if (regions.Count == 1 && regions[0].Start == 0 && regions[0].End == text.Length - 1)
                return GetNumberWithDecimalPlaces(regions[0], canDigitByDigit);

            // Otherwise, replace numeric regions in text
            return ReplaceRegionsInText(text, regions, canDigitByDigit);
        }

        /// <summary>
        /// Combines adjacent regions that are only separated by whitespace.
        /// This allows "one oh two" to become 102 instead of "1 0 2".
        /// </summary>
        private static List<Region> CombineAdjacentRegions(string text, List<Region> regions)
        {
            if (regions.Count <= 1)
                return regions;

            var combined = new List<Region>();
            Region current = null;

            for (int i = 0; i < regions.Count; i++)
            {
                var region = regions[i];

                if (current == null)
                {
                    current = region;
                    continue;
                }

                // Check if regions are adjacent (only whitespace between them)
                int gapStart = current.End + 1;
                int gapEnd = region.Start - 1;
                bool onlyWhitespaceBetween = true;

                if (gapStart <= gapEnd)
                {
                    string gap = text.Substring(gapStart, gapEnd - gapStart + 1);
                    onlyWhitespaceBetween = gap.Trim().Length == 0;
                }

                if (onlyWhitespaceBetween && gapStart <= gapEnd + 1)
                {
                    // Merge regions
                    current.End = region.End;
                    current.Tokens.AddRange(region.Tokens);
                    current.SubRegions.AddRange(region.SubRegions);
                    if (region.HasDecimal)
                        current.HasDecimal = true;
                }
                else
                {
                    // Not adjacent, save current and start new
                    combined.Add(current);
                    current = region;
                }
            }

            // Add the last region
            if (current != null)
                combined.Add(current);

            return combined;
        }

        private static string GetNumberWithDecimalPlaces(Region region, bool canDigitByDigit)
        {
            var number = GetNumber(region, canDigitByDigit);
            return number.decimalPlaces > 0
                ? number.val.ToString($"F{number.decimalPlaces}")
                : number.val.ToString();
        }

        private static bool ShouldUseDigitByDigit(SubRegion subRegion, bool canDigitByDigit)
        {
            // If permission denied, never use digit-by-digit
            if (!canDigitByDigit)
                return false;

            // Always use semantic if magnitude present (thousand, million, etc.)
            if (subRegion.Tokens.Any(t => t.Type == TokenType.Magnitude))
                return false;

            // Always use semantic if marked as Hundred type
            if (subRegion.Tokens.Any(t => t.Type == TokenType.Hundred))
                return false;

            // Detect semantic pattern: Ten followed by Unit (e.g., "eighty five")
            if (HasSemanticTenPlusUnitPattern(subRegion))
                return false;

            // If we have 2+ tokens and permission, use digit-by-digit
            return subRegion.Tokens.Count >= 2;
        }

        private static bool HasSemanticTenPlusUnitPattern(SubRegion subRegion)
        {
            // Check for Ten + Unit pattern (semantic combination)
            for (int i = 0; i < subRegion.Tokens.Count - 1; i++)
            {
                if (subRegion.Tokens[i].Type == TokenType.Ten &&
                    subRegion.Tokens[i + 1].Type == TokenType.Unit)
                {
                    return true;
                }
            }
            return false;
        }


        /// <summary>
        /// Computes the numeric value of a Region by processing its SubRegions.
        /// </summary>
        private static (double val, int decimalPlaces) GetNumber(Region region, bool canDigitByDigit = true)
        {
            List<SubRegion> integerSubs = new();
            List<SubRegion> fractionalSubs = new();

            bool decimalReached = false;

            // separates the region's subregions into two lists, for sub regions before the decimal and subregions after
            foreach (var sub in region.SubRegions)
            {
                if (sub.Type == TokenType.Decimal)
                {
                    decimalReached = true;
                    continue;
                }

                if (!decimalReached) integerSubs.Add(sub);
                else fractionalSubs.Add(sub);
            }

            int integerValue = ComputeSubRegionsValue(integerSubs, canDigitByDigit);

            if (fractionalSubs.Count == 0)
                return (integerValue, 0);

            int fractionalValue = ComputeSubRegionsValue(fractionalSubs, true);

            // Count leading zeros that would be lost in the numeric value
            int leadingZeros = 0;
            bool foundNonZero = false;
            foreach (var sub in fractionalSubs)
            {
                foreach (var token in sub.Tokens)
                {
                    if (Constants.NUMBER.TryGetValue(token.Lower, out var val))
                    {
                        if (!foundNonZero && val == 0)
                            leadingZeros++;
                        else
                            foundNonZero = true;
                    }
                }
            }

            int fractionalDigits = NumUtils.DigitCount(fractionalValue) + leadingZeros;

            double finalValue = integerValue + (fractionalValue / (double)NumUtils.Pow10(fractionalDigits));
            return (finalValue, fractionalDigits);

            /*
            double sum = 0.0;

            bool decimalReached = false;
            int decimalValue = 0;
            int decimalPlaces = 0;

            foreach (var sub in region.SubRegions)
            {
                if (sub.Type == TokenType.Decimal)
                {
                    decimalReached = true;
                    continue;
                }

                if (!decimalReached)
                {
                    sum += ComputeSubRegionValue(sub);
                    continue;
                }

                double subVal = ComputeSubRegionValue(sub);
                int digits = subVal < 10 ? 1 : (int)Math.Floor(Math.Log10(subVal) + 1);
                //// Merge all decimal subregion tokens into one integer
                //foreach (var token in sub.Tokens)
                //{
                //    if (!Constants.NUMBER.TryGetValue(token.Lower, out double v)) continue;
                //    int value = (int)v;

                //    int digits = value < 10 ? 1 : (int)Math.Floor(Math.Log10(value) + 1);

                //    decimalValue = decimalValue * (int)Math.Pow(10, digits) + value; // we'll replace this
                //    decimalPlaces += digits;
                //}
            }

            double scale = 1.0;
            for (int i = 0; i < decimalPlaces; i++)
                scale /= 10.0;

            sum += decimalValue * scale;

            //foreach (var sub in decimals)
            //{
            //    double value = 0;
            //    foreach (var token in sub.Tokens)
            //    {
            //        value += Constants.NUMBER[token.Lower];
            //        Debug.Log($"DECIMALS: The token is {token.Lower}");
            //    }
            //    Debug.Log($"DECIMALS: The final val is {value}");
            //    int digits = value < 10 ? 1 : (int)Math.Floor(Math.Log10(value) + 1);
            //    for (int i = 0; i < digits; i++) scale /= 10.0;

            //    sum += value * scale;
            //    decimalPlaces += digits;
            //}

            return (sum, decimalPlaces);
            */ 
        }

        private static int ComputeSubRegionsValue(List<SubRegion> subRegions, bool canDigitByDigit)
        {
            // Detect if we have multiple single-token subRegions (e.g., "one oh two", "one twenty")
            // These should be concatenated digit-by-digit: 1-0-2 = 102, 1-20 = 120
            bool hasMultipleSingleTokens = canDigitByDigit &&
                subRegions.Count >= 2 &&
                subRegions.All(sr => sr.Tokens.Count == 1);

            int value = 0;

            foreach (var sub in subRegions)
            {
                bool useDigit = hasMultipleSingleTokens || ShouldUseDigitByDigit(sub, canDigitByDigit);

                int subValue = useDigit ? ComputeSubRegionValueDigitByDigit(sub) : ComputeSubRegionValueNormal(sub);

                if (useDigit)
                {
                    int digits = NumUtils.DigitCount(subValue);
                    value = value * NumUtils.Pow10(digits) + subValue;
                }
                else
                {
                    value += subValue;
                }
            }

            return value;
        }

        /// <summary>
        /// Computes the numeric value of a SubRegion
        ///     - Handles HUNDRED and MAGNITUDE multipliers.
        ///     - Skips tokens as needed
        ///     - Preserves order and combination logic
        /// </summary>
        private static int ComputeSubRegionValueNormal(SubRegion sub)
        {
            var tokens = sub.Tokens;
            int count = tokens.Count;
            Debug.Log($"The count for this subReigon is {count}");
            List<double> processedTokens = new();

            for (int i = 0; i < count; i++)
            {
                var token = tokens[i];
                if (!Constants.NUMBER.TryGetValue(token.Lower, out var value))
                    continue;
                Debug.Log($"The value for this token is {value}");
                Debug.Log($"The type for this token is {token.Type}");
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
            return (int)subSum;
        }

        private static int ComputeSubRegionValueDigitByDigit(SubRegion sub)
        {
            int value = 0;

            foreach (var token in sub.Tokens)
            {
                if (!Constants.NUMBER.TryGetValue(token.Lower, out var v))
                    continue;

                int curr = (int)v;
                int digits = NumUtils.DigitCount(curr);
                value = value * NumUtils.Pow10(digits) + curr;
            }

            return value;
        }

        /// <summary>
        /// Replaces numeric regions in text with their numeric values.
        /// </summary>
        private static string ReplaceRegionsInText(string text, List<Region> regions, bool canDigitByDigit)
        {
            var result = text;
            int offset = 0;

            foreach (var region in regions)
            {
                var replacement = GetNumberWithDecimalPlaces(region, canDigitByDigit);
                int length = region.End - region.Start + 1;
                result = StringUtils.Splice(result, region.Start + offset, length, replacement);
                offset += replacement.Length - length;
            }

            return result;
        }

    }
}