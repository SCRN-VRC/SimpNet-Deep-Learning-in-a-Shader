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
                uint2 px = _Buffer_TexelSize.zw * i.uv.xy;
                float col = _Buffer.Load(uint3(px, 0)).x;

                // 15 FPS
                float4 timer = LoadValue(_Buffer, txTimer);
                timer.x += unity_DeltaTime;

                if (timer.x < 0.0667)
                {
                    StoreValue(txTimer, timer, col, px);
                    return col;
                }
                else timer.x = 0.0;

                [branch]
                if (insideArea(txL1Area, px))
                {
                    px -= txL1Area.xy;
                    [branch]
                    if (insideArea(txWL1, px))
                    {
                        px -= txWL1.xy;
                        if (_Time.y < 1.0 || _Reset > 0)
                        {
                            // Debugging
                            uint i = (px.x / 9) % 3;
                            uint j = (px.x / 3) % 3;
                            uint k = px.x % 3;
                            uint l = px.y % 32;
                            col.r = i * j * k / (l + 1.0);
                            if (i == 2 && j == 1 && k == 1 && l == 31)
                            {
                                buffer[0] = col.rrrr;
                            }
                        }
                    }
                    else if (insideArea(txBL1, px))
                    {

                    }
                    else if (insideArea(txL1s, px))
                    {

                    }
                    else if (insideArea(txL1a, px))
                    {

                    }
                    else if (insideArea(txL1Max, px))
                    {

                    }
                    else if (insideArea(txL1iMax, px))
                    {

                    }
                }
                else if (insideArea(txL2Area, px))
                {
                    px -= txL2Area.xy;
                    [branch]
                    if (insideArea(txWL2, px))
                    {

                    }
                    else if (insideArea(txBL2, px))
                    {

                    }
                    else if (insideArea(txL2s, px))
                    {

                    }
                    else if (insideArea(txL2a, px))
                    {

                    }
                    else if (insideArea(txL2Max, px))
                    {

                    }
                    else if (insideArea(txL2iMax, px))
                    {

                    }
                }
                else if (insideArea(txL3Area, px))
                {
                    px -= txL3Area.xy;
                    [branch]
                    if (insideArea(txWL1, px))
                    {

                    }
                    else if (insideArea(txBL3, px))
                    {

                    }
                    else if (insideArea(txL3s, px))
                    {

                    }
                    else if (insideArea(txL3a, px))
                    {

                    }
                    else if (insideArea(txL3Max, px))
                    {

                    }
                    else if (insideArea(txL3iMax, px))
                    {

                    }
                }
                else if (insideArea(txFC1Area, px))
                {
                    px -= txFC1Area.xy;
                    [branch]
                    if (insideArea(txWFC1, px))
                    {

                    }
                    else if (insideArea(txBFC1, px))
                    {

                    }
                    else if (insideArea(txFC1s, px))
                    {

                    }
                    else if (insideArea(txFC1a, px))
                    {

                    }
                }
                else if (insideArea(txFC2Area, px))
                {
                    px -= txFC2Area.xy;
                    [branch]
                    if (insideArea(txWFC2, px))
                    {

                    }
                    else if (insideArea(txBFC2, px))
                    {

                    }
                    else if (insideArea(txFC2s, px))
                    {

                    }
                    else if (insideArea(txFC2a, px))
                    {

                    }
                }
                else if (insideArea(txFC3Area, px))
                {
                    px -= txFC3Area.xy;
                    [branch]
                    if (insideArea(txWFC3, px))
                    {

                    }
                    else if (insideArea(txBFC3, px))
                    {

                    }
                    else if (insideArea(txFC3s, px))
                    {

                    }
                    else if (insideArea(txFC3o, px))
                    {

                    }
                }
                else if (insideArea(txB4Area, px))
                {
                    px -= txB4Area.xy;
                    [branch]
                    if (insideArea(txDBFC3_h, px))
                    {

                    }
                    else if (insideArea(txDBFC3, px))
                    {

                    }
                    else if (insideArea(txDWFC3_h, px))
                    {

                    }
                    else if (insideArea(txDWFC3, px))
                    {

                    }
                    else if (insideArea(txDBFC2_h, px))
                    {

                    }
                    else if (insideArea(txDBFC2, px))
                    {

                    }
                    else if (insideArea(txDWFC2_h, px))
                    {

                    }
                    else if (insideArea(txDWFC2, px))
                    {

                    }
                    else if (insideArea(txDBFC1_h, px))
                    {

                    }
                    else if (insideArea(txDBFC1, px))
                    {

                    }
                    else if (insideArea(txDWFC1_h, px))
                    {

                    }
                    else if (insideArea(txDWFC1, px))
                    {

                    }
                }
                else if (insideArea(txB3Area, px))
                {
                    px -= txB3Area.xy;
                    [branch]
                    if (insideArea(txEMax3, px))
                    {

                    }
                    else if (insideArea(txEL3, px))
                    {

                    }
                    else if (insideArea(txDbL3_h, px))
                    {

                    }
                    else if (insideArea(txDbL3, px))
                    {

                    }
                    else if (insideArea(txDiL3, px))
                    {

                    }
                    else if (insideArea(txDwL3_h, px))
                    {

                    }
                    else if (insideArea(txDwL3, px))
                    {

                    }
                }
                else if (insideArea(txB2Area, px))
                {
                    px -= txB2Area.xy;
                    [branch]
                    if (insideArea(txPadL3, px))
                    {

                    }
                    else if (insideArea(txEL2Max, px))
                    {

                    }
                    else if (insideArea(txEL2, px))
                    {

                    }
                    else if (insideArea(txDbL2_h, px))
                    {

                    }
                    else if (insideArea(txDbL2, px))
                    {

                    }
                    else if (insideArea(txDwL2_h, px))
                    {

                    }
                    else if (insideArea(txDwL2, px))
                    {

                    }
                }
                else if (insideArea(txB1Area, px))
                {
                    px -= txB1Area.xy;
                    [branch]
                    if (insideArea(txPadL2, px))
                    {

                    }
                    else if (insideArea(txEL1Max, px))
                    {

                    }
                    else if (insideArea(txEL1, px))
                    {

                    }
                    else if (insideArea(txDiL1, px))
                    {

                    }
                    else if (insideArea(txDbL1_h, px))
                    {

                    }
                    else if (insideArea(txDbL1, px))
                    {

                    }
                    else if (insideArea(txDwL1_h, px))
                    {

                    }
                    else if (insideArea(txDwL1, px))
                    {

                    }
                }
                return col;
            }
            ENDCG
        }
    }
}
