using System;
using UnityEngine;
using WordsToNumbers;

public class UtilsTest : MonoBehaviour
{
    void Start()
    {
        try
        {
            TestWordsToNumbersAPI();
            //RunDoubleParserTests();
            //RunChainedTests();

            Debug.Log("<color=green>ALL TESTS PASSED SUCCESSFULLY ✔</color>");
        }
        catch (Exception e)
        {
            Debug.LogError("Test run failed with exception: " + e);
        }
    }

    private void TestWordsToNumbersAPI()
    {
        string name = "Word";
        string input;
        string expectedOutput;
        string output;
        int testNumber = 0;

        // --- Test 1 ---
        input = "My blood glucose level is two thousand and fifty six point seventy seven.";
        expectedOutput = "My blood glucose level is 2056.77.";
        output = WordsToNumbersAPI.Convert(input);
        testNumber += 1;
        CheckAndLogTest(name, testNumber, input, expectedOutput, output);

        // --- Test 2 ---
        input = "He is two hundred and one years old.";
        expectedOutput = "He is 201 years old.";
        output = WordsToNumbersAPI.Convert(input);
        testNumber += 1;
        CheckAndLogTest(name, testNumber, input, expectedOutput, output);

        // --- Test 3 ---
        input = "The value is fifty six.";
        expectedOutput = "The value is 56.";
        output = WordsToNumbersAPI.Convert(input);
        testNumber += 1;
        CheckAndLogTest(name, testNumber, input, expectedOutput, output);

        // --- Test 4 ---
        input = "My score is ninety nine point zero one.";
        expectedOutput = "My score is 99.01.";
        output = WordsToNumbersAPI.Convert(input);
        testNumber += 1;
        CheckAndLogTest(name, testNumber, input, expectedOutput, output);

        // --- Test 5 ---
        input = "zero point five is small.";
        expectedOutput = "0.5 is small.";
        output = WordsToNumbersAPI.Convert(input);
        testNumber += 1;
        CheckAndLogTest(name, testNumber, input, expectedOutput, output);

        // --- Test 6 ---
        input = "My level is seven four point oh three.";
        expectedOutput = "My level is 74.03.";
        output = WordsToNumbersAPI.Convert(input);
        testNumber += 1;
        CheckAndLogTest(name, testNumber, input, expectedOutput, output);

        // --- Test 7: Semantic pattern (Ten+Unit) ---
        input = "My reading is eighty five.";
        expectedOutput = "My reading is 85.";
        output = WordsToNumbersAPI.Convert(input);
        testNumber += 1;
        CheckAndLogTest(name, testNumber, input, expectedOutput, output);

        // --- Test 8: Digit-by-digit pattern ---
        input = "The code is eight five.";
        expectedOutput = "The code is 85.";
        output = WordsToNumbersAPI.Convert(input);
        testNumber += 1;
        CheckAndLogTest(name, testNumber, input, expectedOutput, output);

        // --- Test 9: Multiple leading zeros in decimal ---
        input = "The value is five point zero zero one.";
        expectedOutput = "The value is 5.001.";
        output = WordsToNumbersAPI.Convert(input);
        testNumber += 1;
        CheckAndLogTest(name, testNumber, input, expectedOutput, output);

        // --- Test 10: Just decimal (no integer part) ---
        input = "Only point five remains.";
        expectedOutput = "Only 0.5 remains.";
        output = WordsToNumbersAPI.Convert(input);
        testNumber += 1;
        CheckAndLogTest(name, testNumber, input, expectedOutput, output);

        // --- Test 11: Large consecutive units ---
        input = "Code is one oh two oh three.";
        expectedOutput = "Code is 10203.";
        output = WordsToNumbersAPI.Convert(input);
        testNumber += 1;
        CheckAndLogTest(name, testNumber, input, expectedOutput, output);

        // --- Test 12: Magnitude with decimal ---
        input = "Weight is one thousand point five.";
        expectedOutput = "Weight is 1000.5.";
        output = WordsToNumbersAPI.Convert(input);
        testNumber += 1;
        CheckAndLogTest(name, testNumber, input, expectedOutput, output);

        // --- Test 13: Hundred with decimal ---
        input = "Total is three hundred point two five.";
        expectedOutput = "Total is 300.25.";
        output = WordsToNumbersAPI.Convert(input);
        testNumber += 1;
        CheckAndLogTest(name, testNumber, input, expectedOutput, output);

        // --- Test 14: Teens with decimal ---
        input = "Amount is fifteen point seven.";
        expectedOutput = "Amount is 15.7.";
        output = WordsToNumbersAPI.Convert(input);
        testNumber += 1;
        CheckAndLogTest(name, testNumber, input, expectedOutput, output);

        // --- Test 15: Multiple zeros before non-zero ---
        input = "Precision is zero point zero zero five.";
        expectedOutput = "Precision is 0.005.";
        output = WordsToNumbersAPI.Convert(input);
        testNumber += 1;
        CheckAndLogTest(name, testNumber, input, expectedOutput, output);

        // --- Test 16: Complex combination ---
        input = "Result is nine hundred ninety nine point nine nine.";
        expectedOutput = "Result is 999.99.";
        output = WordsToNumbersAPI.Convert(input);
        testNumber += 1;
        CheckAndLogTest(name, testNumber, input, expectedOutput, output);

        // --- Test 17: Cross-boundary - unit + ten pattern ---
        input = "Reading is one twenty point two thirty.";
        expectedOutput = "Reading is 120.230.";
        output = WordsToNumbersAPI.Convert(input);
        testNumber += 1;
        CheckAndLogTest(name, testNumber, input, expectedOutput, output);

        // --- Test 18: Cross-boundary - three consecutive units ---
        input = "The PIN is one two three.";
        expectedOutput = "The PIN is 123.";
        output = WordsToNumbersAPI.Convert(input);
        testNumber += 1;
        CheckAndLogTest(name, testNumber, input, expectedOutput, output);

        // --- Test 19: Cross-boundary - zeros in middle ---
        input = "Year is two oh oh five.";
        expectedOutput = "Year is 2005.";
        output = WordsToNumbersAPI.Convert(input);
        testNumber += 1;
        CheckAndLogTest(name, testNumber, input, expectedOutput, output);

        // --- Test 20: Cross-boundary - multiple trailing zeros ---
        input = "Count is one oh oh oh.";
        expectedOutput = "Count is 1000.";
        output = WordsToNumbersAPI.Convert(input);
        testNumber += 1;
        CheckAndLogTest(name, testNumber, input, expectedOutput, output);

        // --- Test 21: Cross-boundary - unit then teen ---
        input = "Number is five twelve.";
        expectedOutput = "Number is 512.";
        output = WordsToNumbersAPI.Convert(input);
        testNumber += 1;
        CheckAndLogTest(name, testNumber, input, expectedOutput, output);

        // --- Test 22: Cross-boundary - decimal with many digits ---
        input = "Value is one two point three four five.";
        expectedOutput = "Value is 12.345.";
        output = WordsToNumbersAPI.Convert(input);
        testNumber += 1;
        CheckAndLogTest(name, testNumber, input, expectedOutput, output);

        // --- Test 23: Cross-boundary - long integer sequence ---
        input = "Code is seven eight nine.";
        expectedOutput = "Code is 789.";
        output = WordsToNumbersAPI.Convert(input);
        testNumber += 1;
        CheckAndLogTest(name, testNumber, input, expectedOutput, output);

        // --- Test 24: Cross-boundary - unit + teen ---
        input = "ID is one eleven.";
        expectedOutput = "ID is 111.";
        output = WordsToNumbersAPI.Convert(input);
        testNumber += 1;
        CheckAndLogTest(name, testNumber, input, expectedOutput, output);

        // --- Test 25: Cross-boundary - long decimal sequence ---
        input = "Precise value is one two three point four five six seven.";
        expectedOutput = "Precise value is 123.4567.";
        output = WordsToNumbersAPI.Convert(input);
        testNumber += 1;
        CheckAndLogTest(name, testNumber, input, expectedOutput, output);

        // --- Test 26: Cross-boundary - mixed tens and units ---
        input = "Temperature is three forty five.";
        expectedOutput = "Temperature is 345.";
        output = WordsToNumbersAPI.Convert(input);
        testNumber += 1;
        CheckAndLogTest(name, testNumber, input, expectedOutput, output);

        // --- Test 27: Cross-boundary - complex with teens ---
        input = "Score is nine eighty seven point six five four.";
        expectedOutput = "Score is 987.654.";
        output = WordsToNumbersAPI.Convert(input);
        testNumber += 1;
        CheckAndLogTest(name, testNumber, input, expectedOutput, output);

        // --- Test 28: Cross-boundary - all zeros ---
        input = "Zero is zero zero zero.";
        expectedOutput = "0 is 0.";
        output = WordsToNumbersAPI.Convert(input);
        testNumber += 1;
        CheckAndLogTest(name, testNumber, input, expectedOutput, output);

        // --- Test 29: Cross-boundary - leading zero in integer ---
        input = "Value is zero five six.";
        expectedOutput = "Value is 56.";
        output = WordsToNumbersAPI.Convert(input);
        testNumber += 1;
        CheckAndLogTest(name, testNumber, input, expectedOutput, output);

        // --- Test 30: Cross-boundary - original requirement example ---
        input = "Blood sugar is one twenty point two thirty three.";
        expectedOutput = "Blood sugar is 120.233.";
        output = WordsToNumbersAPI.Convert(input);
        testNumber += 1;
        CheckAndLogTest(name, testNumber, input, expectedOutput, output);

        // --- Test 31 ---
        input = "My blood glucose level is one oh two point seven six.";
        expectedOutput = "My blood glucose level is 102.76.";
        output = WordsToNumbersAPI.Convert(input);
        testNumber += 1;
        CheckAndLogTest(name, testNumber, input, expectedOutput, output);

        // --- Test 32 ---
        input = "My blood glucose level is two five six point twelve.";
        expectedOutput = "My blood glucose level is 256.12.";
        output = WordsToNumbersAPI.Convert(input);
        testNumber += 1;
        CheckAndLogTest(name, testNumber, input, expectedOutput, output);

        Debug.Log("<color=cyan>Word Number Parser Tests Passed.</color>");
    }

    /*
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
    */

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
        else
        {
            Debug.Log($"{name} Test {number} SUCEEDED!\nInput: {input} \nExpected: {expectedOutput} \nOutput: {output}");
        }
    }

}
