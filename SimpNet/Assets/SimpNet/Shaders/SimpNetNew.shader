Shader "SimpNet/SimpNetNew"
{
    Properties
    {
        _CamIn ("Cam Input", 2D) = "black" {}
        _Buffer ("Buffer", 2D) = "black" {}
        _InitWeights ("Initial Weights", 2D) = "black" {}
        _TargetClass ("Target Class #", Int) = 0
        _Reset ("Reset Weights", Int) = 0
        _Stop ("Stop Propagation", Int) = 0
        _Train ("Train Network", Float) = 0
        _MaxDist ("Max Distance", Float) = 0.02
    }
    SubShader
    {
        Tags { "Queue"="Overlay+1" "ForceNoShadowCasting"="True" "IgnoreProjector"="True" }
        ZWrite Off
        ZTest Always
        Cull Off


        Pass
        {
            Lighting Off
            SeparateSpecular Off
            Fog { Mode Off }
            
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 5.0

            #include "UnityCG.cginc"
            #include "Includes/SimpNetLayout.cginc"
            #include "Includes/SimpNetFuncs.cginc"

            RWStructuredBuffer<float4> buffer : register(u1);
            Texture2D<float4> _CamIn;
            Texture2D<float> _Buffer;
            Texture2D<float> _InitWeights;
            float4 _Buffer_TexelSize;
            float _MaxDist;
            float _Train;
            int _TargetClass;
            int _Stop;
            int _Reset;

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float3 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = float4(v.uv * 2 - 1, 0, 1);
                #ifdef UNITY_UV_STARTS_AT_TOP
                v.uv.y = 1-v.uv.y;
                #endif
                o.uv.xy = UnityStereoTransformScreenSpaceTex(v.uv);
                o.uv.z = (distance(_WorldSpaceCameraPos,
                    mul(unity_ObjectToWorld, float4(0,0,0,1)).xyz) > _MaxDist ||
                    !unity_OrthoParams.w) ?
                    -1 : 1;
                return o;
            }

            float frag (v2f i) : SV_Target
            {
                clip(i.uv.z);
                int2 px = _Buffer_TexelSize.zw * i.uv.xy;
                float col = _Buffer.Load(int3(px, 0)).x;

                // 15 FPS
                float4 timer = LoadValue(_Buffer, txTimer);
                timer.x += unity_DeltaTime;

                if (timer.x < 0.0667)
                {
                    StoreValue(txTimer, timer, col, px);
                    return col;
                }
                else timer.x = 0.0;

                return col;
            }
            ENDCG
        }
    }
}
