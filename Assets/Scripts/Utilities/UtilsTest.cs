using System;
using UnityEngine;

public class UtilsTest : MonoBehaviour
{
    void Start()
    {
        try
        {
            RunWordNumberParserTests();
            RunDoubleParserTests();
            RunChainedTests();

            Debug.Log("<color=green>ALL TESTS PASSED SUCCESSFULLY ✔</color>");
        }
        catch (Exception e)
        {
            Debug.LogError("Test run failed with exception: " + e);
            throw;
        }
    }

    private void RunWordNumberParserTests()
    {
        Debug.Assert(
            WordNumberParser.ConvertWordNumbersInSentence("My blood glucose level is one hundred and thirty four point twelve")
            == "My blood glucose level is 134.12",
            "Word Test 1 Failed"
        );

        Debug.Assert(
            WordNumberParser.ConvertWordNumbersInSentence("He is two hundred and one years old")
            == "He is 201 years old",
            "Word Test 2 Failed"
        );

        Debug.Assert(
            WordNumberParser.ConvertWordNumbersInSentence("The value is fifty six")
            == "The value is 56",
            "Word Test 3 Failed"
        );

        Debug.Assert(
            WordNumberParser.ConvertWordNumbersInSentence("My score is ninety nine point zero one")
            == "My score is 99.01",
            "Word Test 4 Failed"
        );

        Debug.Assert(
            WordNumberParser.ConvertWordNumbersInSentence("zero point five is small")
            == "0.5 is small",
            "Word Test 5 Failed"
        );
    }

    private void RunDoubleParserTests()
    {
        Debug.Assert(
            DoubleFromStringParser.ParseDoubleFromString("Value is 42") == 42,
            "Double Test 1 Failed"
        );

        Debug.Assert(
            DoubleFromStringParser.ParseDoubleFromString("Glucose is 134.12 mg/dL") == 134.12,
            "Double Test 2 Failed"
        );

        Debug.Assert(
            DoubleFromStringParser.ParseDoubleFromString("Values 56 then 78") == 56,
            "Double Test 3 Failed"
        );

        Debug.Assert(
            DoubleFromStringParser.ParseDoubleFromString("No numbers here") == null,
            "Double Test 4 Failed"
        );

        Debug.Assert(
            DoubleFromStringParser.ParseDoubleFromString(null) == null,
            "Double Test 5 Failed"
        );
    }

    private void RunChainedTests()
    {
        // Full pipeline test: words → numbers → double
        string s = "My glucose is one hundred and nine point five";

        Debug.Assert(Utils.GetDoubleFromNonnumericalSentence(s) == 109.5, "Chained Test 1 Failed");
  
    }
}
