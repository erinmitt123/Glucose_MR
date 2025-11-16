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
    public double glucoseVal=200;
    private bool isTypeOne=true;

    //values retreived from database
    public double[] secureMLValues;

    //values that are set based on calculations
    public Sprite[] emojiDatabase;
    public TMP_Text grade;
    public Image emoji; // drag object with GlucoseData onto this

    public void GetGlucose()
    {
        //glucoseVal = ApplicationManager.Instance.glucoseLevel;
        //isTypeOne=ApplicationManager.Instance.isTypeOne;
    }
    public void SetValues(double[] values)
    {
     secureMLValues = values;
     Debug.Log("FoodInfo received " + values.Length + " values.");
     food.text = "Food: "+secureMLValues[0].ToString();
     carbs.text = "Carbs: " + secureMLValues[1].ToString();
     fats.text = "Fats: " + secureMLValues[2].ToString();
     fiber.text = "Fiber: " + secureMLValues[3].ToString();
     protein.text = "Protein: " + secureMLValues[4].ToString();
     sugar.text = "Sugar: " + secureMLValues[5].ToString();
        if (secureMLValues[6] < 0){
            secureMLValues[6] = 0;
        }
     addedSugars.text = "Added Sugars: " + secureMLValues[6].ToString();
     perUnit.text = "Per 100g"; //+ secureMLValues[7].ToString();
            double score = CalculateScore(secureMLValues, glucoseVal);

        if (isTypeOne)
        {
            TypeOneHelper(score);
        }
        else
        {
            string letterGrade = CalculateGrade(score);
            ChooseColor(letterGrade);
            grade.text = "Score: " + score.ToString() + "%, " + letterGrade;
        }

     foreach (double value in secureMLValues)
     {
         Debug.Log(value);
     }
 }

    private void TypeOneHelper(double score)
    {
        if (glucoseVal < 70) {
            if (score > 90)
            {
                grade.text = "Great Low Snack!";
                grade.color = new Color32(0, 200, 0, 255); // green
                emoji.sprite = emojiDatabase[0];
            }
            else
            {
                grade.text = "Consume a Low Snack First";
                grade.color = new Color32(255, 0, 0, 255); // red
                emoji.sprite = emojiDatabase[3];
            }
        }
        else if (score < 60)
        {
            grade.text = "Warning! This may cause a spike.";
            grade.color = new Color32(255, 180, 0, 255); // yellow/orange
            emoji.sprite = emojiDatabase[4];
        }
        else {
            grade.text = "Nice choice!";
            grade.color = new Color32(80, 180, 0, 255);
            emoji.sprite = emojiDatabase[1];
        }
    }

    //this method calculates the score for the user
    public int CalculateScore(double[] secureMLValues, double glucoseVal)
    {
        double score;
        if (glucoseVal < 70)
        {
            //these vals are hard coded to their column because I am being fast for the hackathons sake sorry
            score = 13 + 10 * (secureMLValues[1] + secureMLValues[5] + secureMLValues[6]) / (secureMLValues[2] + secureMLValues[3] + secureMLValues[4]);
        }
        else if (glucoseVal < 100)
        {
            score = 90 + 0.3 * ((secureMLValues[2] + secureMLValues[3] + secureMLValues[4]) - (secureMLValues[1] + secureMLValues[5] + (secureMLValues[6])));
        }
        else
        {
            score = (-glucoseVal/2)+140 + 0.3 * ((secureMLValues[2] + secureMLValues[3] + secureMLValues[4]) - (secureMLValues[1] + secureMLValues[5] + (secureMLValues[6])));
        }
        Debug.Log("score"+score);
        if (score > 100)
            score = 100;

        return (int)score;
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
    public void ChooseColor(string gradeText)
    {        
            // Get first character as uppercase (A, B, C, D, F)
            char letter = char.ToUpper(gradeText[0]);

            switch (letter)
            {
                case 'A':
                    grade.color = new Color32(0, 200, 0, 255); // green
                    emoji.sprite = emojiDatabase[0];
                    break;

                case 'B':
                    grade.color = new Color32(80, 180, 0, 255);
                    emoji.sprite = emojiDatabase[1];
                    break;

                case 'C':
                    grade.color = new Color32(255, 180, 0, 255); // yellow/orange
                    emoji.sprite = emojiDatabase[2];   
                    break;

                case 'D':
                    grade.color = new Color32(255, 120, 0, 255);
                    emoji.sprite = emojiDatabase[2];
                    break;

                case 'F':
                    grade.color = new Color32(255, 0, 0, 255); // red
                    emoji.sprite = emojiDatabase[3];
                    break;

                default:
                    grade.color = Color.white;
                    break;
            }
        }
}
