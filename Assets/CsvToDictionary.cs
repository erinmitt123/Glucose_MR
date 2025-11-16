using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class CsvToDictionary : MonoBehaviour
{
    public TextAsset csvFile;  // drag your .csv into inspector
    public FoodInfo foodInfo;


    public Dictionary<string, int> ParseToDictionary()
    {
        Dictionary<string, int> dict = new Dictionary<string, int>();

        string[] lines = csvFile.text.Split('\n');

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

    void Start()
    {
        var map = ParseToDictionary();
        foodInfo.SetDict(map);
        Debug.Log("Loaded " + map.Count + " entries");
    }
}

