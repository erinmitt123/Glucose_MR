using System.Collections.Generic;

namespace WordsToNumbers
{

    #region -----ENUMS-----

    public enum Action
    {
        SKIP = 0,
        ADD = 1,
        START_NEW_REGION = 2,
        NOPE = 3
    }

    public enum TokenType
    {
        Unit = 0,
        Ten = 1,
        Magnitude = 2,
        Decimal = 3,
        Hundred = 4
    }

    #endregion

    #region -----CLASSES-----

    public class Token
    {
        public int Start;
        public int End;
        public string Value;
        public string Lower;
        public TokenType? Type;
    }

    public class SubRegion
    {
        public List<Token> Tokens = new();
        public TokenType Type;
    }

    public class Region
    {
        public int Start;
        public int End;
        public bool HasDecimal;
        public List<Token> Tokens = new();
        public List<SubRegion> SubRegions = new();
    }

    #endregion

}