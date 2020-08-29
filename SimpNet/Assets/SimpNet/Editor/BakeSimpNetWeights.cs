#if UNITY_EDITOR
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using System.IO;
using System.Text.RegularExpressions;
using UnityEngine.UI;

[ExecuteInEditMode]
public class BakeSimpNetWeights : EditorWindow {
    
    public TextAsset source0;
    public TextAsset source1;
    public TextAsset source2;
    public TextAsset source3;

    string SavePath = "Assets/SimpNet/Weights/WeightsTex.asset";

    public int[,] savePos =
    {
        // Weights
        { 547, 128 },    // Kern1
        { 258, 128 },   // Kern2
        { 0, 0 },       // Kern3
        { 0, 128 },     // W1
        { 129, 128 },   // W2
        { 258, 192 },     // W3
        // Bias
        { 574, 128 },     // BK1
        { 546, 128 },     // BK2
        { 576, 0 },     // BK3
        { 128, 128 },     // BW1
        { 257, 128 },     // BW2
        { 386, 192 }      // BW3
    };

    [MenuItem("Tools/SCRN/Bake SimpNet")]
    static void Init()
    {
        var window = GetWindowWithRect<BakeSimpNetWeights>(new Rect(0, 0, 400, 250));
        window.Show();
    }

    void OnGUI()
    {
        GUILayout.Label("Bake SimpNet", EditorStyles.boldLabel);
        EditorGUILayout.BeginVertical();
        EditorGUILayout.BeginHorizontal();
        source0 = (TextAsset) EditorGUILayout.ObjectField("SimpNet Weights0 (.txt):", source0, typeof(TextAsset), false);
        EditorGUILayout.EndHorizontal();
        source1 = (TextAsset) EditorGUILayout.ObjectField("SimpNet Weights1 (.txt):", source1, typeof(TextAsset), false);
        source2 = (TextAsset) EditorGUILayout.ObjectField("SimpNet Weights2 (.txt):", source2, typeof(TextAsset), false);
        source3 = (TextAsset) EditorGUILayout.ObjectField("SimpNet Weights3 (.txt):", source3, typeof(TextAsset), false);
        EditorGUILayout.EndVertical();

        if (GUILayout.Button("Bake!")) {
            OnGenerateTexture();
        }
    }

    int ExtractFloats(Texture2D tex, TextAsset srcIn, Regex rgWs, int width, int wIndex,
        int offX, int offY)
    {
        Match mWs = rgWs.Match(srcIn.text);
        string strWs = mWs.Groups[0].Value;

        if (strWs.Length == 0) {
            ShowNotification(new GUIContent("Wrong file format"));
            Debug.Log("Index: " + wIndex + " failed to capture correct values");
            return 0;
        }

        // Grab floats
        Regex rgFloats = new Regex("[-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?");
        MatchCollection fvals = rgFloats.Matches(strWs);

        //Debug.Log("Count: " + fvals.Count);
        
        int c = 0;
        foreach (Match fValMatches in fvals)
        {
            int x = c % width;
            int y = c / width;

            CaptureCollection fValCapture = fValMatches.Captures;
            float stf = float.Parse(fValCapture[0].Value);
            tex.SetPixel(offX + savePos[wIndex, 0] + x, offY + savePos[wIndex, 1] + y,
                new Color(stf, 0.0f, 0.0f, 0.0f));
            // tex.SetPixel(savePos[wIndex, 0] + x, savePos[wIndex, 1] + y,
            //     new Color((wIndex + 1.0f) / 12.0f, 0.0f, 0.0f, 0.0f));
            c++;
        }

        return c;
    }

    void ExtractFromText(Texture2D tex, TextAsset srcIn, int offX, int offY)
    {
        // Kern1 weights
        Regex rgWs = new Regex("(?<=kern1:)[\\s\\S]*(?=bias1:)");
        ExtractFloats(tex, srcIn, rgWs, 27, 0, offX, offY);

        // Kern2 weights
        rgWs = new Regex("(?<=kern2:)[\\s\\S]*(?=bias2:)");
        ExtractFloats(tex, srcIn, rgWs, 288, 1, offX, offY);

        // Kern3 weights
        rgWs = new Regex("(?<=kern3:)[\\s\\S]*(?=bias3:)");
        ExtractFloats(tex, srcIn, rgWs, 576, 2, offX, offY);

        // W1 weights
        rgWs = new Regex("(?<=w1:)[\\s\\S]*(?=biasw1:)");
        ExtractFloats(tex, srcIn, rgWs, 128, 3, offX, offY);

        // W2 weights
        rgWs = new Regex("(?<=w2:)[\\s\\S]*(?=biasw2:)");
        ExtractFloats(tex, srcIn, rgWs, 128, 4, offX, offY);

        // W3 weights
        rgWs = new Regex("(?<=w3:)[\\s\\S]*(?=biasw3:)");
        ExtractFloats(tex, srcIn, rgWs, 128, 5, offX, offY);

        // bias1 weights
        rgWs = new Regex("(?<=bias1:)[\\s\\S]*(?=kern2:)");
        ExtractFloats(tex, srcIn, rgWs, 1, 6, offX, offY);

        // bias2 weights
        rgWs = new Regex("(?<=bias2:)[\\s\\S]*(?=kern3:)");
        ExtractFloats(tex, srcIn, rgWs, 1, 7, offX, offY);

        // bias3 weights
        rgWs = new Regex("(?<=bias3:)[\\s\\S]*?(?=w1:)");
        ExtractFloats(tex, srcIn, rgWs, 1, 8, offX, offY);

        // biasw1 weights
        rgWs = new Regex("(?<=biasw1:)[\\s\\S]*?(?=w2:)");
        ExtractFloats(tex, srcIn, rgWs, 1, 9, offX, offY);

        // biasw2 weights
        rgWs = new Regex("(?<=biasw2:)[\\s\\S]*?(?=w3:)");
        ExtractFloats(tex, srcIn, rgWs, 1, 10, offX, offY);
        
        // biasw3 weights
        rgWs = new Regex("(?<=biasw3:)[\\s\\S]*");
        ExtractFloats(tex, srcIn, rgWs, 1, 11, offX, offY);
    }

    void OnGenerateTexture()
    {
        const int width = 577;
        const int height = 256;
        Texture2D tex = new Texture2D(width, height * 4, TextureFormat.RFloat, false);
        tex.wrapMode = TextureWrapMode.Clamp;
        tex.filterMode = FilterMode.Point;
        tex.anisoLevel = 1;

        if (source0 != null)
            ExtractFromText(tex, source0, 0, 0);
        if (source1 != null)
            ExtractFromText(tex, source1, 0, height);
        if (source2 != null)
            ExtractFromText(tex, source2, 0, height * 2);
        if (source3 != null)
            ExtractFromText(tex, source3, 0, height * 3);

        AssetDatabase.CreateAsset(tex, SavePath);
        AssetDatabase.SaveAssets();

        ShowNotification(new GUIContent("Done"));
    }
}

#endif