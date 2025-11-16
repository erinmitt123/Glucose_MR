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


        foreach (double value in secureMLValues)
        {
            Debug.Log(value);
        }
    }

}
