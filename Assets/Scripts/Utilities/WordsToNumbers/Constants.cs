using System.Collections.Generic;
using System.Linq;

namespace WordsToNumbers
{
    public static class Constants
    {
        // Used by the parser, simple rules for checking if a token can be added to the end of a subregion
        // included for cleaner code and quicker lookup at the cost of negligible storage
        public static readonly Dictionary<(TokenType, TokenType), bool> SimpleMatchRules = new()
        {
            // Magnitude rules
            { (TokenType.Magnitude, TokenType.Unit), true },
            { (TokenType.Magnitude, TokenType.Ten), true },
            { (TokenType.Magnitude, TokenType.Magnitude), true },

            // Ten rules without implied hundreds handled separately
            { (TokenType.Ten, TokenType.Unit), true },
        };

        #region -----NUMBER DICTIONARIES-----

        public static readonly Dictionary<string, double> UNIT = new()
        {
            {"oh", 0 },{"zero",0},{"first",1},{"one",1},{"second",2},{"two",2},{"third",3},{"three",3},
            {"fourth",4},{"four",4},{"fifth",5},{"five",5},{"sixth",6},{"six",6},
            {"seventh",7},{"seven",7},{"eighth",8},{"eight",8},{"ninth",9},{"nine",9},
            {"tenth",10},{"ten",10},{"eleventh",11},{"eleven",11},{"twelfth",12},{"twelve",12},
            {"thirteenth",13},{"thirteen",13},{"fourteenth",14},{"fourteen",14},
            {"fifteenth",15},{"fifteen",15},{"sixteenth",16},{"sixteen",16},
            {"seventeenth",17},{"seventeen",17},{"eighteenth",18},{"eighteen",18},
            {"nineteenth",19},{"nineteen",19},{"a",1}
        };

        public static readonly Dictionary<string, double> TEN = new()
        {
            {"twenty",20},{"thirty",30},{"forty",40},{"fifty",50},
            {"sixty",60},{"seventy",70},{"eighty",80},{"ninety",90}
        };

        public static readonly Dictionary<string, double> MAGNITUDE = new()
        {
            {"hundred",100},{"thousand",1_000},{"million",1_000_000},
            {"billion",1_000_000_000},{"trillion",1_000_000_000_000}
        };

        public static readonly Dictionary<string, double> NUMBER = UNIT
        .Concat(TEN)
        .Concat(MAGNITUDE)
        .ToDictionary(k => k.Key, v => v.Value);

        #endregion

        #region -----SPECIAL STRINGS HASHSETS-----

        public static readonly HashSet<string> JOINERS = new() { "and" };
        public static readonly HashSet<string> DECIMALS = new() { "point", "dot" };
        public static readonly HashSet<string> BLACKLIST_SINGULAR_WORDS = new() { "a" };

        public static readonly HashSet<string> PUNCTUATION = new()
        {
            ".", ",", "\\", "#", "!", "$", "%", "^", "&", "/", "*", ";", ":", "{", "}",
            "=", "-", "_", "`", "~", "(", ")", " "
        };

        #endregion

    }
}