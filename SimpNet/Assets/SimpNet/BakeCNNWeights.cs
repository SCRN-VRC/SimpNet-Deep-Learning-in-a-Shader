#if UNITY_EDITOR
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using System.IO;
using System.Text.RegularExpressions;
using UnityEngine.UI;

[ExecuteInEditMode]
public class BakeCNNWeights : EditorWindow {
    
    public TextAsset source;
    string SavePath1 = "Assets/SimpNet/weights.asset";

    [MenuItem("Tools/SCRN/Bake CNN Weights")]
    static void Init()
    {
        var window = GetWindowWithRect<BakeCNNWeights>(new Rect(0, 0, 400, 250));
        window.Show();
    }

    void OnGUI()
    {
        GUILayout.Label("Bake CNN Weights", EditorStyles.boldLabel);
        EditorGUILayout.BeginHorizontal();
        source = (TextAsset) EditorGUILayout.ObjectField("CNN Weights (.txt):", source, typeof(TextAsset), false);
        EditorGUILayout.EndHorizontal();

        if (GUILayout.Button("Bake!")) {
           
            if (source == null)
                ShowNotification(new GUIContent("Select the .txt output file"));
            else
                OnGenerateTexture();
        }
    }

    void OnGenerateTexture()
    {
        // Kern1 weights
        Regex rgWs = new Regex("(?<=kern1:)[\\s\\S]*(?=bias1:)");

        Match mWs = rgWs.Match(source.text);
        string strWs = mWs.Groups[0].Value;

        if (strWs.Length == 0) {
            ShowNotification(new GUIContent("Wrong file format"));
            return;
        }

        // Grab floats
        Regex rgFloats = new Regex("[-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?");
        MatchCollection fvals = rgFloats.Matches(strWs);
    }
}

#endif