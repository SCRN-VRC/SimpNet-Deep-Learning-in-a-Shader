﻿Shader "Unlit/UnlitAlpha"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _Cutout ("Cutout", Float) = 0.5
        _DiscardOrtho ("Discard Orthographic", Int) = 0
    }
    SubShader
    {
        Tags { "RenderQueue"="Geometry" "RenderType"="Geometry" }

        Pass
        {
            Cull Off

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            sampler2D _MainTex;
            float4 _MainTex_ST;
            float _Cutout;
            int _DiscardOrtho;
            
            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                if (_DiscardOrtho > 0 && unity_OrthoParams.w) discard;
                // sample the texture
                fixed4 col = tex2D(_MainTex, i.uv);
                col.rgb *= col.a;
                clip(col.a - _Cutout);
                return col;
            }
            ENDCG
        }
    }
}
