Shader "SimpNet/ClassifyOut"
{
    Properties
    {
        _NNBuffer ("Neural Net Buffer", 2D) = "white" {}
        _Labels ("Labels", 2D) = "white" {}
        _Test ("test", Vector) = (0, 0, 0, 0)
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 5.0

            #include "UnityCG.cginc"
            #include "Includes/SimpNetLayout.cginc"

            #define labelWH             int2(140, 28)
            #define labelOffset         int4(15, 468, 280, 56)

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

            static int toIndex[6] = { 4, 5, 2, 3, 0, 1 };
            
            //RWStructuredBuffer<float4> buffer : register(u1);
            Texture2D<float> _NNBuffer;
            Texture2D<float> _Labels;
            float4 _Test;

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            float4 frag (v2f i) : SV_Target
            {
                float3 col = 0.0;
                float2 fuv = fmod(i.uv, float2(0.5, 0.3333)) * float2(2.0, 3.0);
                int2 uv_id = floor(i.uv * float2(2.0, 3.0));
                uv_id.x = uv_id.x + uv_id.y * 2;

                float2 top6[6] = {
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                };

                for (int i = 0; i < 12; i++) {
                    float2 curScore = float2(_NNBuffer.Load(int3(txFC3Area.xy + txFC3o.xy + int2(0, i), 0)).x, i);
                    for (int j = 0; j < 6; j++) {
                        float2 prevTop = curScore.x > top6[j].x ? top6[j] : curScore;
                        top6[j] = curScore.x > top6[j].x ? curScore : top6[j];
                        curScore = curScore.x > top6[j].x ? top6[j] : prevTop;
                    }
                }

                int2 pos = labelOffset.xy + fuv * labelWH;
                pos.x += (int(top6[toIndex[uv_id.x]].y) % 2) * labelOffset.z;
                pos.y -= (int(top6[toIndex[uv_id.x]].y) / 2) * labelOffset.w;

                col.rgb = lerp(1.0, float3(0.0, 0.5, 1.0), top6[toIndex[uv_id.x]].x > fuv.x);
                col.rgb = lerp(0.0, col.rgb, _Labels.Load(int3(pos, 0)).r);

                return float4(col, 1.0);
            }
            ENDCG
        }
    }
}
