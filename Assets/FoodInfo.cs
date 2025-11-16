using System;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

//attach this to each food when identified
public class FoodInfo : MonoBehaviour
{
    //values that are set based on database values
    public TMP_Text food;
    public TMP_Text perUnit;
    public TMP_Text carbs;
    public TMP_Text fats;
    public TMP_Text fiber;
    public TMP_Text protein;
    public TMP_Text sugar;
    public TMP_Text addedSugars;
    public double glucoseVal;

    //values retreived from database
    public double[] secureMLValues;

    //values that are set based on calculations
    public Sprite[] emojiDatabase;
    public TMP_Text grade;
    public Image emoji;

    //activate this canvas when data is received about each 
    public void Awake()
    {
        food.text = "Food: "+secureMLValues[0].ToString();
        carbs.text = "Carbs: " + secureMLValues[1].ToString();
        fats.text = "Fats: " + secureMLValues[2].ToString();
        fiber.text = "Fiber: " + secureMLValues[3].ToString();
        protein.text = "Protein: " + secureMLValues[4].ToString();
        sugar.text = "Sugar: " + secureMLValues[5].ToString();
        addedSugars.text = "Added Sugars: " + secureMLValues[6].ToString();
        perUnit.text += "Per 100g"; //+ secureMLValues[7].ToString();
        double score=CalculateScore(secureMLValues, glucoseVal);
        string letterGrade = CalculateGrade(score);
        grade.text = "Score " + score.ToString() + "%, "+ letterGrade;
        foreach (double value in secureMLValues)
        {
            Debug.Log(value);
        }
    }

    //this method calculates the score for the user
    public double CalculateScore(double[] secureMLValues, double glucoseVal)
    {
        double score;
        if (glucoseVal < 70)
        {
            //these vals are hard coded to their column because I am being fast for the hackathons sake sorry
            score = 15 + 10 * (secureMLValues[1] + secureMLValues[5] + secureMLValues[6]) / (secureMLValues[2] + secureMLValues[3] + secureMLValues[4]);
        }
        else if (glucoseVal < 100)
        {
            score = 90 + 0.3 * ((secureMLValues[2] + secureMLValues[3] + secureMLValues[4]) - (secureMLValues[1] + secureMLValues[5] + (secureMLValues[6] + 1)));

        }
        else
        {
            score = (glucoseVal/2)+140 + 0.3 * ((secureMLValues[2] + secureMLValues[3] + secureMLValues[4]) - (secureMLValues[1] + secureMLValues[5] + (secureMLValues[6] + 1)));

        }
        Debug.Log("score"+score);
        if (score > 100)
            score = 100;

        return score;
    }
    public string CalculateGrade(double score)
    {
            return score switch
            {
                >= 97 and <= 100 => "A+",
                >= 93 and < 97 => "A",
                >= 90 and < 93 => "A-",

                >= 87 and < 90 => "B+",
                >= 83 and < 87 => "B",
                >= 80 and < 83 => "B-",

                >= 77 and < 80 => "C+",
                >= 73 and < 77 => "C",
                >= 70 and < 73 => "C-",

                >= 67 and < 70 => "D+",
                >= 63 and < 67 => "D",
                >= 60 and < 63 => "D-",

                >= 0 and < 60 => "F",

                _ => "Invalid score"
            };
        }
}
