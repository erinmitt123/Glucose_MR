using System;
using System.Linq;
using UnityEngine;

public class CsvToDoubleArray : MonoBehaviour
{
    public TextAsset csvFile;
    public FoodInfo foodInfo;

    void Start()
    {
        double[][] data = ParseCsv(csvFile.text);
        foodInfo.SetValues(data[7]);
    }

    public double[][] ParseCsv(string csvText)
    {
        return csvText
            .Split('\n')                                      // each row
            .Where(line => !string.IsNullOrWhiteSpace(line))  // skip blank lines
            .Select(line =>
                line.Split(',')                               // split row into columns
                    .Select(v => double.Parse(v.Trim()))      // convert each cell to double
                    .ToArray()
            )
            .ToArray();
    }
}
