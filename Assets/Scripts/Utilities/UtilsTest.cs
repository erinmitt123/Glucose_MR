using System;
using UnityEngine;
using UnityEngine.Windows;

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
        }
    }

    private void RunWordNumberParserTests()
    {
        string name = "Word";
        string input;
        string expectedOutput;
        string output;
        int testNumber = 0;

        // --- Test 1 ---
        input = "My blood glucose level is one hundred and thirty four point twelve.";
        expectedOutput = "My blood glucose level is 134.12.";
        output = WordNumberParser.ConvertWordNumbersInSentence(input);
        testNumber += 1;
        CheckAndLogTest(name, testNumber, input, expectedOutput, output);

        // --- Test 2 ---
        input = "He is two hundred and one years old.";
        expectedOutput = "He is 201 years old.";
        output = WordNumberParser.ConvertWordNumbersInSentence(input);
        testNumber += 1;
        CheckAndLogTest(name, testNumber, input, expectedOutput, output);

        // --- Test 3 ---
        input = "The value is fifty six.";
        expectedOutput = "The value is 56.";
        output = WordNumberParser.ConvertWordNumbersInSentence(input);
        testNumber += 1;
        CheckAndLogTest(name, testNumber, input, expectedOutput, output);

        // --- Test 4 ---
        input = "My score is ninety nine point zero one.";
        expectedOutput = "My score is 99.01.";
        output = WordNumberParser.ConvertWordNumbersInSentence(input);
        testNumber += 1;
        CheckAndLogTest(name, testNumber, input, expectedOutput, output);

        // --- Test 5 ---
        input = "zero point five is small.";
        expectedOutput = "0.5 is small.";
        output = WordNumberParser.ConvertWordNumbersInSentence(input);
        testNumber += 1;
        CheckAndLogTest(name, testNumber, input, expectedOutput, output);

        // --- Test 6 ---
        input = "My level is seven four point oh three.";
        expectedOutput = "My level is 74.03.";
        output = WordNumberParser.ConvertWordNumbersInSentence(input);
        testNumber += 1;
        CheckAndLogTest(name, testNumber, input, expectedOutput, output);

        Debug.Log("<color=cyan>Word Number Parser Tests Passed.</color>");
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

        Debug.Log("<color=cyan>Double Parser Tests Passed.</color>");
    }

    private void RunChainedTests()
    {
        // Full pipeline test: words → numbers → double
        string s = "My glucose is one hundred and nine point five";

        Debug.Assert(Utils.GetDoubleFromNonnumericalSentence(s) == 109.5, "Chained Test 1 Failed");

        Debug.Log("<color=cyan>Double From Nonnumerical Sentence Tests Passed.</color>");
    }

    private void CheckAndLogTest(string name, int number, string input, string expectedOutput, string output)
    {
        if (output != expectedOutput)
        {
            throw new Exception(
                $"{name} Test {number} Failed.\nInput: {input} \nExpected: {expectedOutput} \nOutput: {output}"
            );
        }
    }

}
