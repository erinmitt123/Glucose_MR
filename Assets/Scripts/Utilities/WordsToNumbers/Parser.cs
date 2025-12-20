using System.Collections.Generic;
using System.Text.RegularExpressions;
using UnityEditor;

namespace WordsToNumbers
{
    internal static class Parser
    {
        private static readonly Regex TokenSplitRegex = new("(\\w+|\\s|[^\\w])", RegexOptions.Compiled);

        public static List<Region> Parse(string text, bool fuzzy = false, bool impliedHundreds = false)
        {
            var tokens = Tokenize(text, fuzzy);
            return MatchRegions(tokens, impliedHundreds);
        }

        #region -----TOKENS-----

        public static List<Token> Tokenize(string text, bool fuzzy)
        {
            var tokens = new List<Token>();
            int index = 0;

            foreach (var chunk in TokenSplitRegex.Split(text))
            {
                if (string.IsNullOrEmpty(chunk)) continue;

                string unfuzzy = fuzzy ? Fuzzy.Match(chunk) : chunk;
                var lower = unfuzzy.ToLower();

                tokens.Add(new Token
                {
                    Start = index,
                    End = index + unfuzzy.Length - 1,
                    Value = unfuzzy,
                    Lower = lower,
                    Type = GetTokenType(lower)
                });

                index += chunk.Length;
            }

            return tokens;
        }

        private static TokenType? GetTokenType(string chunk) => chunk switch
        {
            _ when Constants.UNIT.ContainsKey(chunk) => TokenType.Unit,
            _ when Constants.TEN.ContainsKey(chunk) => TokenType.Ten,
            _ when Constants.MAGNITUDE.ContainsKey(chunk) => TokenType.Magnitude,
            _ when Constants.DECIMALS.Contains(chunk) => TokenType.Decimal,
            _ => null
        };

        private static bool CheckBlacklist(List<Token> tokens) 
            => tokens.Count == 1 && Constants.BLACKLIST_SINGULAR_WORDS.Contains(tokens[0].Lower);

        #endregion

        #region -----REGION PARSING-----

        private static List<Region> MatchRegions(List<Token> tokens, bool impliedHundreds)
        {
            List<Region> regions = new();
            if (CheckBlacklist(tokens))
                return regions;

            Region currRegion = null;
            
            foreach (var token in tokens)
            {
                currRegion = CheckIfTokenFitsRegion(currRegion, token, impliedHundreds) switch
                {
                    Action.SKIP => currRegion,
                    Action.ADD when currRegion != null => AddTokenToRegion(currRegion, token),
                    Action.START_NEW_REGION => CreateNewRegion(regions, token),
                    _ => null
                };
            }

            foreach (var region in regions)
                region.SubRegions = BuildSubRegions(region, impliedHundreds);

            return regions;
        }

        // Helper: Adds a token to an existing region
        private static Region AddTokenToRegion(Region region, Token token)
        {
            region.End = token.End;
            region.Tokens.Add(token);
            if (token.Type == TokenType.Decimal)
                region.HasDecimal = true;

            return region;
        }

        // Helper: Creates a new region and adds it to the list
        private static Region CreateNewRegion(List<Region> regions, Token token)
        {
            var region = new Region
            {
                Start = token.Start,
                End = token.End,
                Tokens = new List<Token> { token },
                HasDecimal = token.Type == TokenType.Decimal
            };
            regions.Add(region);
            return region;
        }

        private static Action CheckIfTokenFitsRegion(Region region, Token token, bool impliedHundreds)
        {
            if (Constants.PUNCTUATION.Contains(token.Lower) || Constants.JOINERS.Contains(token.Lower))
                return Action.SKIP;

            if (Constants.DECIMALS.Contains(token.Lower))
            {
                if (region == null || region.Tokens.Count == 0)
                    return Action.START_NEW_REGION;

                if (!region.HasDecimal)
                    return Action.ADD;
            }

            bool isNumberWord = Constants.NUMBER.ContainsKey(token.Lower);
            if (isNumberWord)
                return region != null && CanAddTokenToEndOfRegion(region, token, impliedHundreds) ? Action.ADD : Action.START_NEW_REGION;

            return Action.NOPE;
        }

        private static bool CanAddTokenToEndOfRegion(Region region, Token currToken, bool impliedHundreds)
        {
            var tokens = region.Tokens;
            if (tokens.Count == 0)
                return true;

            var prevToken = tokens[^1];

            if (impliedHundreds)
                return true;

            return !(
                (prevToken.Type == TokenType.Unit && currToken.Type == TokenType.Unit && !region.HasDecimal)
                || (prevToken.Type == TokenType.Unit && currToken.Type == TokenType.Ten)
                || (prevToken.Type == TokenType.Ten && currToken.Type == TokenType.Ten)
            );
        }

        #endregion

        #region -----SUBREGION PARSING-----

        private static List<SubRegion> BuildSubRegions(Region region, bool impliedHundreds)
        {
            var subs = new List<SubRegion>();
            SubRegion currSub = null;

            for (int i = region.Tokens.Count - 1; i >= 0; i--)
            {
                var token = region.Tokens[i];

                var (action, type, isHundred) = CheckIfTokenFitsSubRegion(currSub, token, impliedHundreds);
                token.Type = isHundred ? TokenType.Hundred : token.Type;

                switch (action)
                {
                    case Action.ADD:
                        currSub.Type = type;
                        currSub.Tokens.Add(token);
                        break;
                    case Action.START_NEW_REGION:
                        currSub = new SubRegion 
                        { 
                            Tokens = new List<Token> { token }, 
                            Type = token.Type.Value 
                        };
                        subs.Add(currSub);
                        break;                        
                }
            }

            subs.Reverse();
            foreach (var sub in subs) 
                sub.Tokens.Reverse();

            return subs;
        }

        private static (Action action, TokenType type, bool isHundred) CheckIfTokenFitsSubRegion(SubRegion subRegion, Token token, bool impliedHundreds)
        {
            var (type, isHundred) = GetSubRegionType(subRegion, token);

            return subRegion != null && CanAddTokenToEndOfSubRegion(subRegion, token, impliedHundreds) 
                    ? (Action.ADD, type, isHundred)
                    : (Action.START_NEW_REGION, type, isHundred);
        }

        private static (TokenType type, bool isHundred) GetSubRegionType(SubRegion subRegion, Token curr)
        {
            if (subRegion == null)
                return (curr.Type.Value, false);

            var prev = subRegion.Tokens[0];

            bool isHundred =
                (prev.Type == TokenType.Ten && (curr.Type == TokenType.Unit || curr.Type == TokenType.Ten)) ||
                (
                    prev.Type == TokenType.Unit && (curr.Type == TokenType.Unit || 
                        (
                            curr.Type == TokenType.Ten 
                            && Constants.UNIT[prev.Lower] > 9
                        ))
                ) 
                || (subRegion.Type == TokenType.Magnitude && prev.Type == TokenType.Ten && curr.Type == TokenType.Unit);

            if (subRegion.Type == TokenType.Magnitude)
                return (TokenType.Magnitude, isHundred);

            return isHundred ? (TokenType.Hundred, true) : (curr.Type.Value, false);
        }

        private static bool CanAddTokenToEndOfSubRegion(SubRegion subRegion, Token curr, bool impliedHundreds)
        {
            if (subRegion.Tokens.Count == 0)
                return true;

            var prev = subRegion.Tokens[0];

            if (prev.Type == TokenType.Ten && curr.Type == TokenType.Ten)
                return impliedHundreds;

            if (impliedHundreds && subRegion.Type == TokenType.Magnitude &&
                (
                    (prev.Type == TokenType.Ten && curr.Type == TokenType.Unit) 
                    || (prev.Type == TokenType.Unit && curr.Type == TokenType.Ten)
                ))
                return true;

            return Constants.SimpleMatchRules.TryGetValue((prev.Type.Value, curr.Type.Value), out var result) 
                && result;
        }
            

        #endregion

    }
}