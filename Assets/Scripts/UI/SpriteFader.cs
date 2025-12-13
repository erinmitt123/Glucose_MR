using System.Collections;
using UnityEngine;

[RequireComponent(typeof(SpriteRenderer))]
public class SpriteFader : MonoBehaviour
{
    [SerializeField] private float fadeDuration = 0.2f;


    private SpriteRenderer spriteRenderer;
    private Coroutine fadeRoutine;
    private float minVisibleAlpha = 0.01f;

    void Awake()
    {
        spriteRenderer = GetComponent<SpriteRenderer>();
        SetAlpha(0f);
        spriteRenderer.enabled = true;
    }

    public void FadeIn()
    {
        if (fadeRoutine != null)
            StopCoroutine(fadeRoutine);

        fadeRoutine = StartCoroutine(FadeInNextFrame());
    }

    private IEnumerator FadeInNextFrame()
    {
        SetAlpha(minVisibleAlpha);
        yield return null;
        fadeRoutine = StartCoroutine(FadeTo(1f));
    }

    public void FadeOut()
    {
        if (fadeRoutine != null)
            StopCoroutine(fadeRoutine);

        fadeRoutine = StartCoroutine(FadeTo(0f));
    }

    private IEnumerator FadeTo(float targetAlpha)
    {
        float startAlpha = spriteRenderer.color.a;
        float elapsed = 0f;
        transform.localScale *= 0.95f;

        while (elapsed < fadeDuration)
        {
            elapsed += Time.deltaTime;

            float t = elapsed / fadeDuration;
            t = Mathf.SmoothStep(0f, 1f, t);
            float alpha = Mathf.Lerp(startAlpha, targetAlpha, t);

            SetAlpha(alpha);
            yield return null;
        }

        SetAlpha(targetAlpha);
    }

    private void SetAlpha(float alpha)
    {
        Color c = spriteRenderer.color;
        c.a = alpha;
        spriteRenderer.color = c;
    }
}
