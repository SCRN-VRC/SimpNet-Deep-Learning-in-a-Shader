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
                        }
                    }
                    else if (insideArea(txBL1, px))
                    {
                        px -= txBL1.xy;
                        if (_Time.y < 1.0 || _Reset > 0)
                        {
                            // Debugging
                            uint i = px.y;
                            col.r = i / 32.0 - 0.5;
                        }
                    }
                    else if (insideArea(txL1s, px))
                    {
                        px -= txL1s.xy;
                        uint i = (px.x / 32) % 32;
                        uint j = px.x % 32;
                        uint k = px.y % 32;
                        uint i0 = i * 2, i1 = i0 + 1, i2 = i0 + 2;
                        uint j0 = j * 2, j1 = j0 + 1, j2 = j0 + 2;

                        float sum = 0.0;
                        
                        for (uint l = 0; l < 3; l++) {
                            // sum += (_CamIn.Load(int3(j0, 64 - i0, 0))[l]) * getWL1(_Buffer, uint4(0, 0, l, k));
                            // sum += (_CamIn.Load(int3(j0, 64 - i1, 0))[l]) * getWL1(_Buffer, uint4(0, 1, l, k));
                            // sum += (_CamIn.Load(int3(j0, 64 - i2, 0))[l]) * getWL1(_Buffer, uint4(0, 2, l, k));
                            // sum += (_CamIn.Load(int3(j1, 64 - i0, 0))[l]) * getWL1(_Buffer, uint4(1, 0, l, k));
                            // sum += (_CamIn.Load(int3(j1, 64 - i1, 0))[l]) * getWL1(_Buffer, uint4(1, 1, l, k));
                            // sum += (_CamIn.Load(int3(j1, 64 - i2, 0))[l]) * getWL1(_Buffer, uint4(1, 2, l, k));
                            // sum += (_CamIn.Load(int3(j2, 64 - i0, 0))[l]) * getWL1(_Buffer, uint4(2, 0, l, k));
                            // sum += (_CamIn.Load(int3(j2, 64 - i1, 0))[l]) * getWL1(_Buffer, uint4(2, 1, l, k));
                            // sum += (_CamIn.Load(int3(j2, 64 - i2, 0))[l]) * getWL1(_Buffer, uint4(2, 2, l, k));
                       
                            sum += testImage(i0, j0, l) * getWL1(_Buffer, uint4(0, 0, l, k));
                            sum += testImage(i0, j1, l) * getWL1(_Buffer, uint4(0, 1, l, k));
                            sum += testImage(i0, j2, l) * getWL1(_Buffer, uint4(0, 2, l, k));
                            sum += testImage(i1, j0, l) * getWL1(_Buffer, uint4(1, 0, l, k));
                            sum += testImage(i1, j1, l) * getWL1(_Buffer, uint4(1, 1, l, k));
                            sum += testImage(i1, j2, l) * getWL1(_Buffer, uint4(1, 2, l, k));
                            sum += testImage(i2, j0, l) * getWL1(_Buffer, uint4(2, 0, l, k));
                            sum += testImage(i2, j1, l) * getWL1(_Buffer, uint4(2, 1, l, k));
                            sum += testImage(i2, j2, l) * getWL1(_Buffer, uint4(2, 2, l, k));
                        }

                        sum += getBL1(_Buffer, k);
                        col.r = sum;
                    }
                    else if (insideArea(txL1a, px))
                    {
                        px -= txL1a.xy;
                        uint i = (px.x / 32) % 32;
                        uint j = px.x % 32;
                        uint k = px.y % 32;

                        col.r = afn(getL1s(_Buffer, uint3(i, j, k)));
                    }
                    else if (insideArea(txL1Max, px))
                    {
                        px -= txL1Max.xy;
                        uint i = (px.x / 16) % 16;
                        uint j = px.x % 16;
                        uint k = px.y % 32;
                        uint i0 = i * 2, i1 = i0 + 1;
                        uint j0 = j * 2, j1 = j0 + 1;

                        float m = getL1a(_Buffer, uint3(i0, j0, k));
                        m = max(m, getL1a(_Buffer, uint3(i0, j1, k)));
                        m = max(m, getL1a(_Buffer, uint3(i1, j0, k)));
                        m = max(m, getL1a(_Buffer, uint3(i1, j1, k)));
                        col.r = m;
                    }
                    else if (insideArea(txL1iMax, px))
                    {
                        px -= txL1iMax.xy;
                        uint i = (px.x / 16) % 16;
                        uint j = px.x % 16;
                        uint k = px.y % 32;
                        uint i0 = i * 2, i1 = i0 + 1;
                        uint j0 = j * 2, j1 = j0 + 1;

                        float buf;
                        float m = getL1a(_Buffer, int3(i0, j0, k));
                        col.r = i0 * 32 + j0;
                        m = max(m, buf = getL1a(_Buffer, uint3(i0, j1, k)));
                        col.r = (m == buf) ? i0 * 32 + j1 : col.r;
                        m = max(m, buf = getL1a(_Buffer, uint3(i1, j0, k)));
                        col.r = (m == buf) ? i1 * 32 + j0 : col.r;
                        m = max(m, buf = getL1a(_Buffer, uint3(i1, j1, k)));
                        col.r = (m == buf) ? i1 * 32 + j1 : col.r;
                    }
                }
                else if (insideArea(txL2Area, px))
                {
                    px -= txL2Area.xy;
                    [branch]
                    if (insideArea(txWL2, px))
                    {
                        px -= txWL2.xy;
                        if (_Time.y < 1.0 || _Reset > 0)
                        {
                            // Debugging
                            uint i = (px.x / 96) % 3;
                            uint j = (px.x / 32) % 3;
                            uint k = px.x % 32;
                            uint l = px.y % 64;
                            col.r = (i + j + k + l) / 1000.0;
                        }
                    }
                    else if (insideArea(txBL2, px))
                    {
                        px -= txBL2.xy;
                        if (_Time.y < 1.0 || _Reset > 0)
                        {
                            // Debugging
                            uint i = px.y % 64;
                            col.r = 1.0 - (i / 64.0) - 0.5;
                        }
                    }
                    else if (insideArea(txL2s, px))
                    {
                        px -= txL2s.xy;
                        uint i = (px.x / 14) % 14;
                        uint j = px.x % 14;
                        uint k = px.y % 64;
                        uint i0 = i, i1 = i + 1, i2 = i + 2;
                        uint j0 = j, j1 = j + 1, j2 = j + 2;

                        float sum = 0.0;
                        
                        for (uint l = 0; l < 32; l++) {
                            sum += getL1Max(_Buffer, uint3(i0, j0, l)) * getWL2(_Buffer, uint4(0, 0, l, k));
                            sum += getL1Max(_Buffer, uint3(i0, j1, l)) * getWL2(_Buffer, uint4(0, 1, l, k));
                            sum += getL1Max(_Buffer, uint3(i0, j2, l)) * getWL2(_Buffer, uint4(0, 2, l, k));
                            sum += getL1Max(_Buffer, uint3(i1, j0, l)) * getWL2(_Buffer, uint4(1, 0, l, k));
                            sum += getL1Max(_Buffer, uint3(i1, j1, l)) * getWL2(_Buffer, uint4(1, 1, l, k));
                            sum += getL1Max(_Buffer, uint3(i1, j2, l)) * getWL2(_Buffer, uint4(1, 2, l, k));
                            sum += getL1Max(_Buffer, uint3(i2, j0, l)) * getWL2(_Buffer, uint4(2, 0, l, k));
                            sum += getL1Max(_Buffer, uint3(i2, j1, l)) * getWL2(_Buffer, uint4(2, 1, l, k));
                            sum += getL1Max(_Buffer, uint3(i2, j2, l)) * getWL2(_Buffer, uint4(2, 2, l, k));
                        }

                        sum += getBL2(_Buffer, k);
                        col.r = sum;
                    }
                    else if (insideArea(txL2a, px))
                    {
                        px -= txL2a.xy;
                        uint i = (px.x / 14) % 14;
                        uint j = px.x % 14;
                        uint k = px.y % 64;

                        col.r = afn(getL2s(_Buffer, uint3(i, j, k)));
                    }
                    else if (insideArea(txL2Max, px))
                    {
                        px -= txL2Max.xy;
                        uint i = (px.x / 7) % 7;
                        uint j = px.x % 7;
                        uint k = px.y % 64;
                        uint i0 = i * 2, i1 = i0 + 1;
                        uint j0 = j * 2, j1 = j0 + 1;

                        float m = getL2a(_Buffer, uint3(i0, j0, k));
                        m = max(m, getL2a(_Buffer, uint3(i0, j1, k)));
                        m = max(m, getL2a(_Buffer, uint3(i1, j0, k)));
                        m = max(m, getL2a(_Buffer, uint3(i1, j1, k)));
                        col.r = m;
                    }
                    else if (insideArea(txL2iMax, px))
                    {
                        px -= txL2iMax.xy;
                        uint i = (px.x / 7) % 7;
                        uint j = px.x % 7;
                        uint k = px.y % 64;
                        uint i0 = i * 2, i1 = i0 + 1;
                        uint j0 = j * 2, j1 = j0 + 1;

                        float buf;
                        float m = getL2a(_Buffer, int3(i0, j0, k));
                        col.r = i0 * 14 + j0;
                        m = max(m, buf = getL2a(_Buffer, uint3(i0, j1, k)));
                        col.r = (m == buf) ? i0 * 14 + j1 : col.r;
                        m = max(m, buf = getL2a(_Buffer, uint3(i1, j0, k)));
                        col.r = (m == buf) ? i1 * 14 + j0 : col.r;
                        m = max(m, buf = getL2a(_Buffer, uint3(i1, j1, k)));
                        col.r = (m == buf) ? i1 * 14 + j1 : col.r;
                    }
                }
                else if (insideArea(txL3Area, px))
                {
                    px -= txL3Area.xy;
                    [branch]
                    if (insideArea(txWL3, px))
                    {
                        px -= txWL3.xy;
                        if (_Time.y < 1.0 || _Reset > 0)
                        {
                            // Debugging
                            uint i = (px.x / 192) % 3;
                            uint j = (px.x / 64) % 3;
                            uint k = px.x % 64;
                            uint l = px.y % 128;
                            col.r = (i + j) / float(k + l + 1.0);
                        }
                    }
                    else if (insideArea(txBL3, px))
                    {
                        px -= txBL3.xy;
                        if (_Time.y < 1.0 || _Reset > 0)
                        {
                            // Debugging
                            uint i = px.y % 128;
                            col.r = (i / 128.0) - 0.5;
                        }
                    }
                    else if (insideArea(txL3s, px))
                    {
                        px -= txL3s.xy;
                        uint i = (px.x / 3) % 3;
                        uint j = px.x % 3;
                        uint k = px.y % 128;
                        uint i0 = i * 2, i1 = i0 + 1, i2 = i0 + 2;
                        uint j0 = j * 2, j1 = j0 + 1, j2 = j0 + 2;

                        float sum = 0.0;

                        for (uint l = 0; l < 64; l++) {
                            sum += getL2Max(_Buffer, uint3(i0, j0, l)) * getWL3(_Buffer, uint4(0, 0, l, k));
                            sum += getL2Max(_Buffer, uint3(i0, j1, l)) * getWL3(_Buffer, uint4(0, 1, l, k));
                            sum += getL2Max(_Buffer, uint3(i0, j2, l)) * getWL3(_Buffer, uint4(0, 2, l, k));
                            sum += getL2Max(_Buffer, uint3(i1, j0, l)) * getWL3(_Buffer, uint4(1, 0, l, k));
                            sum += getL2Max(_Buffer, uint3(i1, j1, l)) * getWL3(_Buffer, uint4(1, 1, l, k));
                            sum += getL2Max(_Buffer, uint3(i1, j2, l)) * getWL3(_Buffer, uint4(1, 2, l, k));
                            sum += getL2Max(_Buffer, uint3(i2, j0, l)) * getWL3(_Buffer, uint4(2, 0, l, k));
                            sum += getL2Max(_Buffer, uint3(i2, j1, l)) * getWL3(_Buffer, uint4(2, 1, l, k));
                            sum += getL2Max(_Buffer, uint3(i2, j2, l)) * getWL3(_Buffer, uint4(2, 2, l, k));
                        }

                        sum += getBL3(_Buffer, k);
                        col.r = sum;
                    }
                    else if (insideArea(txL3a, px))
                    {
                        px -= txL3a.xy;
                        uint i = (px.x / 3) % 3;
                        uint j = px.x % 3;
                        uint k = px.y % 128;

                        col.r = afn(getL3s(_Buffer, uint3(i, j, k)));
                    }
                    else if (insideArea(txL3Max, px))
                    {
                        px -= txL3Max.xy;
                        uint k = px.y % 128;

                        float m = getL3a(_Buffer, uint3(0, 0, k));
                        for (uint i = 0; i < 3; i++) {
                            for (uint j = 0; j < 3; j++) {
                                m = max(m, getL3a(_Buffer, uint3(i, j, k)));
                            }
                        }
                        col.r = m;
                    }
                    else if (insideArea(txL3iMax, px))
                    {
                        px -= txL3iMax.xy;
                        uint k = px.y % 128;

                        float buff;
                        float m = getL3a(_Buffer, uint3(0, 0, k));
                        col.r = 0.0;
                        for (uint i = 0; i < 3; i++) {
                            for (uint j = 0; j < 3; j++) {
                                m = max(m, buff = getL3a(_Buffer, uint3(i, j, k)));
                                col.r = (m == buff) ? (i * 3 + j) : col.r;
                            }
                        }
                    }
                }
                else if (insideArea(txFC1Area, px))
                {
                    px -= txFC1Area.xy;
                    [branch]
                    if (insideArea(txWFC1, px))
                    {
                        px -= txWFC1.xy;
                        if (k == 0)
                        {
                            buffer[0] = col.rrrr;
                        }
                    }
                    else if (insideArea(txBFC1, px))
                    {
                        px -= txBFC1.xy;
                    }
                    else if (insideArea(txFC1s, px))
                    {
                        px -= txFC1s.xy;
                    }
                    else if (insideArea(txFC1a, px))
                    {
                        px -= txFC1a.xy;
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
