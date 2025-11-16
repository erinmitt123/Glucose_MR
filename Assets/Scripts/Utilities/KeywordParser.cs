public static class KeywordParser
{
    public static string FindKeyword(string sentence)
    {
        if (string.IsNullOrEmpty(sentence)) return null;

        string lowerSentence = sentence.ToLowerInvariant();

        foreach (string keyword in IdentifiableObjects.Objects)
        {
            if (string.IsNullOrEmpty(keyword)) 
                continue;

            string lowerKeyword = keyword.ToLowerInvariant();

            if (lowerSentence.Contains(lowerKeyword))
                return keyword;
        }

        return null;
    }
}
