using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class CsvToDoubleArray : MonoBehaviour
{
    public static CsvToDoubleArray Instance { get; private set; }

    public TextAsset csvFile;
    public TextAsset csvFileDict;

    public double[][] nutritionDataArray;

    private void Awake()
    {
        // Enforce singleton pattern
        if (Instance != null && Instance != this)
        {
            Destroy(gameObject);
            return;
        }

        Instance = this;
        DontDestroyOnLoad(gameObject);
    }

    private void Start()
    {
        var map = ParseToDictionary();
        FoodInfo.Instance.SetDict(map);
        Debug.Log("Loaded " + map.Count + " entries");

        nutritionDataArray = ParseCsv(csvFile.text);
        FoodInfo.Instance.UpdateValuesAndDisplay();
    }

    public Dictionary<string, int> ParseToDictionary()
    {
        Dictionary<string, int> dict = new Dictionary<string, int>();

        string[] lines = csvFileDict.text.Split('\n');

        foreach (string line in lines)
        {
            if (string.IsNullOrWhiteSpace(line))
                continue;

            string[] parts = line.Split(',');

            string key = parts[0].Trim();
            int value = int.Parse(parts[1].Trim());

            dict[key] = value;
        }

        return dict;
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
