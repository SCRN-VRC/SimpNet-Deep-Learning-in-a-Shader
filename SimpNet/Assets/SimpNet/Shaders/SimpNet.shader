Shader "SimpNet/SimpNet"
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
            uint _TargetClass;
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
                float timer = LoadValue(_Buffer, txTimer);
                timer += unity_DeltaTime;

                if (timer < 0.0667)
                {
                    StoreValue(txTimer, timer, col, px);
                    return col;
                }
                else timer = 0.0;

                bool initTime = (_Time.y < 1.0) || (_Reset > 0);

                // Layer count
                float lcF = LoadValue(_Buffer, txLC);
                uint lc = _Stop > 0 ? 0 : floor(lcF);

                if (insideArea(txL1Area, px))
                {
                    px -= txL1Area.xy;
                    [branch]
                    if (lc == 1 && insideArea(txWL1, px))
                    {
                        px -= txWL1.xy;
                        uint i = (px.x / 9) % 3;
                        uint j = (px.x / 3) % 3;
                        uint k = px.x % 3;
                        uint l = px.y % 32;
                        if (initTime)
                        {
                            // Debugging
                            //col.r = i * j * k / (l + 1.0);
                            col.r = _InitWeights.Load(uint3(txInitKern1 + px, 0));
                        }
                        float delta = lr * (getDwL1(_Buffer, uint4(i, j, k, l)) /
                            (sqrt(getDwL1_m(_Buffer, uint4(i, j, k, l))) + epsilon));

                        col.r -= ((_Train > 0.0 && !initTime) ? 1.0 : 0.0) * delta;
                    }
                    else if (lc == 1 && insideArea(txBL1, px))
                    {
                        px -= txBL1.xy;
                        uint i = px.y;
                        if (initTime)
                        {
                            // Debugging
                            //col.r = i / 32.0 - 0.5;
                            col.r = _InitWeights.Load(uint3(txInitB1 + px, 0));
                        }
                        float delta = lr * (getDbL1(_Buffer, i) /
                            (sqrt(getDbL1_m(_Buffer, i)) + epsilon));

                        col.r -= ((_Train > 0.0 && !initTime) ? 1.0 : 0.0) * delta;
                    }
                    else if (lc == 2 && insideArea(txL1s, px))
                    {
                        px -= txL1s.xy;
                        uint i = (px.x / 32) % 32;
                        uint j = px.x % 32;
                        uint k = px.y % 32;
                        uint i0 = i * 2, i1 = i0 + 1, i2 = i0 + 2;
                        uint j0 = j * 2, j1 = j0 + 1, j2 = j0 + 2;

                        float sum = 0.0;
                        
                        for (uint l = 0; l < 3; l++) {
                            sum += (_CamIn.Load(int3(j0, 64 - i0, 0))[l]) * getWL1(_Buffer, uint4(0, 0, l, k));
                            sum += (_CamIn.Load(int3(j0, 64 - i1, 0))[l]) * getWL1(_Buffer, uint4(0, 1, l, k));
                            sum += (_CamIn.Load(int3(j0, 64 - i2, 0))[l]) * getWL1(_Buffer, uint4(0, 2, l, k));
                            sum += (_CamIn.Load(int3(j1, 64 - i0, 0))[l]) * getWL1(_Buffer, uint4(1, 0, l, k));
                            sum += (_CamIn.Load(int3(j1, 64 - i1, 0))[l]) * getWL1(_Buffer, uint4(1, 1, l, k));
                            sum += (_CamIn.Load(int3(j1, 64 - i2, 0))[l]) * getWL1(_Buffer, uint4(1, 2, l, k));
                            sum += (_CamIn.Load(int3(j2, 64 - i0, 0))[l]) * getWL1(_Buffer, uint4(2, 0, l, k));
                            sum += (_CamIn.Load(int3(j2, 64 - i1, 0))[l]) * getWL1(_Buffer, uint4(2, 1, l, k));
                            sum += (_CamIn.Load(int3(j2, 64 - i2, 0))[l]) * getWL1(_Buffer, uint4(2, 2, l, k));
                       
                            // sum += testImage(i0, j0, l) * getWL1(_Buffer, uint4(0, 0, l, k));
                            // sum += testImage(i0, j1, l) * getWL1(_Buffer, uint4(0, 1, l, k));
                            // sum += testImage(i0, j2, l) * getWL1(_Buffer, uint4(0, 2, l, k));
                            // sum += testImage(i1, j0, l) * getWL1(_Buffer, uint4(1, 0, l, k));
                            // sum += testImage(i1, j1, l) * getWL1(_Buffer, uint4(1, 1, l, k));
                            // sum += testImage(i1, j2, l) * getWL1(_Buffer, uint4(1, 2, l, k));
                            // sum += testImage(i2, j0, l) * getWL1(_Buffer, uint4(2, 0, l, k));
                            // sum += testImage(i2, j1, l) * getWL1(_Buffer, uint4(2, 1, l, k));
                            // sum += testImage(i2, j2, l) * getWL1(_Buffer, uint4(2, 2, l, k));
                        }

                        sum += getBL1(_Buffer, k);
                        col.r = sum;
                    }
                    else if (lc == 3 && insideArea(txL1a, px))
                    {
                        px -= txL1a.xy;
                        uint i = (px.x / 32) % 32;
                        uint j = px.x % 32;
                        uint k = px.y % 32;

                        col.r = afn(getL1s(_Buffer, uint3(i, j, k)));
                    }
                    else if (lc == 4 && insideArea(txL1Max, px))
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
                    else if (lc == 5 && insideArea(txL1iMax, px))
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
                    if (lc == 6 && insideArea(txWL2, px))
                    {
                        px -= txWL2.xy;
                        uint i = (px.x / 96) % 3;
                        uint j = (px.x / 32) % 3;
                        uint k = px.x % 32;
                        uint l = px.y % 64;
                        if (initTime)
                        {
                            // Debugging
                            //col.r = (i + j + k + l) / 1000.0;
                            col.r = _InitWeights.Load(uint3(txInitKern2 + px, 0));
                        }
                        float delta = lr * (getDwL2(_Buffer, uint4(i, j, k, l)) /
                            (sqrt(getDwL2_m(_Buffer, uint4(i, j, k, l))) + epsilon));

                        col.r -= ((_Train > 0.0 && !initTime) ? 1.0 : 0.0) * delta;
                    }
                    else if (lc == 6 && insideArea(txBL2, px))
                    {
                        px -= txBL2.xy;
                        uint i = px.y % 64;
                        if (initTime)
                        {
                            // Debugging
                            //col.r = 1.0 - (i / 64.0) - 0.5;
                            col.r = _InitWeights.Load(uint3(txInitB2 + px, 0));
                        }
                        float delta = lr * (getDbL2(_Buffer, i) /
                            (sqrt(getDbL2_m(_Buffer, i)) + epsilon));

                        col.r -= ((_Train > 0.0 && !initTime) ? 1.0 : 0.0) * delta;
                    }
                    else if (lc == 7 && insideArea(txL2s, px))
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
                    else if (lc == 8 && insideArea(txL2a, px))
                    {
                        px -= txL2a.xy;
                        uint i = (px.x / 14) % 14;
                        uint j = px.x % 14;
                        uint k = px.y % 64;

                        col.r = afn(getL2s(_Buffer, uint3(i, j, k)));
                    }
                    else if (lc == 9 && insideArea(txL2Max, px))
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
                    else if (lc == 10 && insideArea(txL2iMax, px))
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
                    if (lc == 11 && insideArea(txWL3, px))
                    {
                        px -= txWL3.xy;
                        uint i = (px.x / 192) % 3;
                        uint j = (px.x / 64) % 3;
                        uint k = px.x % 64;
                        uint l = px.y % 128;
                        if (initTime)
                        {
                            // Debugging
                            //col.r = (i + j) / float(k + l + 1.0);
                            col.r = _InitWeights.Load(uint3(txInitKern3 + px, 0));
                        }
                        float delta = lr * (getDwL3(_Buffer, uint4(i, j, k, l)) /
                            (sqrt(getDwL3_m(_Buffer, uint4(i, j, k, l))) + epsilon));

                        col.r -= ((_Train > 0.0 && !initTime) ? 1.0 : 0.0) * delta;
                    }
                    else if (lc == 11 && insideArea(txBL3, px))
                    {
                        px -= txBL3.xy;
                        uint i = px.y % 128;
                        if (initTime)
                        {
                            // Debugging
                            //col.r = (i / 128.0) - 0.5;
                            col.r = _InitWeights.Load(uint3(txInitB3 + px, 0));
                        }
                        float delta = lr * (getDbL3(_Buffer, i) /
                            (sqrt(getDbL3_m(_Buffer, i)) + epsilon));

                        col.r -= ((_Train > 0.0 && !initTime) ? 1.0 : 0.0) * delta;
                    }
                    else if (lc == 12 && insideArea(txL3s, px))
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
                    else if (lc == 13 && insideArea(txL3a, px))
                    {
                        px -= txL3a.xy;
                        uint i = (px.x / 3) % 3;
                        uint j = px.x % 3;
                        uint k = px.y % 128;

                        col.r = afn(getL3s(_Buffer, uint3(i, j, k)));
                    }
                    else if (lc == 14 && insideArea(txL3Max, px))
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
                    else if (lc == 15 && insideArea(txL3iMax, px))
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
                    if (lc == 16 && insideArea(txWFC1, px))
                    {
                        px -= txWFC1.xy;
                        uint i = px.x % 128;
                        uint j = px.y % 128;
                        if (initTime)
                        {
                            // Debugging
                            //col.r = (i + j) / float(i * j + 100000);
                            col.r = _InitWeights.Load(uint3(txInitW1 + px, 0));
                        }
                        float delta = lr * (getDWFC1(_Buffer, uint2(i, j)) /
                            (sqrt(getDWFC1_m(_Buffer, uint2(i, j))) + epsilon));

                        col.r -= ((_Train > 0.0 && !initTime) ? 1.0 : 0.0) * delta;
                    }
                    else if (lc == 16 && insideArea(txBFC1, px))
                    {
                        px -= txBFC1.xy;
                        uint i = px.y % 128;
                        if (initTime)
                        {
                            // Debugging
                            //col.r = i / float(i + 100);
                            col.r = _InitWeights.Load(uint3(txInitBw1 + px, 0));
                        }
                        float delta = lr * (getDBFC1(_Buffer, i) /
                            (sqrt(getDBFC1_m(_Buffer, i)) + epsilon));

                        col.r -= ((_Train > 0.0 && !initTime) ? 1.0 : 0.0) * delta;
                    }
                    else if (lc == 17 && insideArea(txFC1s, px))
                    {
                        px -= txFC1s.xy;
                        uint k = px.y % 128;

                        float sum = 0.0;
                        for (uint l = 0; l < 128; l++) {
                            sum += getL3Max(_Buffer, l) * getWFC1(_Buffer, uint2(l, k));
                        }

                        sum += getBFC1(_Buffer, k);
                        col.r = sum;
                    }
                    else if (lc == 18 && insideArea(txFC1a, px))
                    {
                        px -= txFC1a.xy;
                        uint k = px.y % 128;

                        col.r = afn(getFC1s(_Buffer, k));
                    }
                }
                else if (insideArea(txFC2Area, px))
                {
                    px -= txFC2Area.xy;
                    [branch]
                    if (lc == 19 && insideArea(txWFC2, px))
                    {
                        px -= txWFC2.xy;
                        uint i = px.x % 128;
                        uint j = px.y % 128;
                        if (initTime)
                        {
                            // Debugging
                            //col.r = (i + j) / float(i * j + 200000);
                            col.r = _InitWeights.Load(uint3(txInitW2 + px, 0));
                        }
                        float delta = lr * (getDWFC2(_Buffer, uint2(i, j)) /
                            (sqrt(getDWFC2_m(_Buffer, uint2(i, j))) + epsilon));

                        col.r -= ((_Train > 0.0 && !initTime) ? 1.0 : 0.0) * delta;
                    }
                    else if (lc == 19 && insideArea(txBFC2, px))
                    {
                        px -= txBFC2.xy;
                        uint i = px.y % 128;
                        if (initTime)
                        {
                            // Debugging
                            //col.r = i / float(i + 1000);
                            col.r = _InitWeights.Load(uint3(txInitBw2 + px, 0));
                        }
                        float delta = lr * (getDBFC2(_Buffer, i) /
                            (sqrt(getDBFC2_m(_Buffer, i)) + epsilon));

                        col.r -= ((_Train > 0.0 && !initTime) ? 1.0 : 0.0) * delta;
                    }
                    else if (lc == 20 && insideArea(txFC2s, px))
                    {
                        px -= txFC2s.xy;
                        uint k = px.y % 128;

                        float sum = 0.0;
                        for (uint l = 0; l < 128; l++) {
                            sum += getFC1a(_Buffer, l) * getWFC2(_Buffer, uint2(l, k));
                        }

                        sum += getBFC2(_Buffer, k);
                        col.r = sum;
                    }
                    else if (lc == 21 && insideArea(txFC2a, px))
                    {
                        px -= txFC2a.xy;
                        uint k = px.y % 128;

                        col.r = afn(getFC2s(_Buffer, k));
                    }
                }
                else if (insideArea(txFC3Area, px))
                {
                    px -= txFC3Area.xy;
                    [branch]
                    if (lc == 22 && insideArea(txWFC3, px))
                    {
                        px -= txWFC3.xy;
                        uint i = px.x % 128;
                        uint j = px.y % 12;
                        if (initTime)
                        {
                            // Debugging
                            //col.r = (i + 12 - j) / float(j + 2000);
                            col.r = _InitWeights.Load(uint3(txInitW3 + px, 0));
                        }
                        float delta = lr * (getDWFC3(_Buffer, uint2(i, j)) /
                            (sqrt(getDWFC3_m(_Buffer, uint2(i, j))) + epsilon));

                        col.r -= ((_Train > 0.0 && !initTime) ? 1.0 : 0.0) * delta;
                    }
                    else if (lc == 22 && insideArea(txBFC3, px))
                    {
                        px -= txBFC3.xy;
                        uint i = px.y % 12;
                        if (initTime)
                        {
                            // Debugging
                            //col.r = i / 11.0 - 0.5;
                            col.r = _InitWeights.Load(uint3(txInitBw3 + px, 0));
                        }
                        float delta = lr * (getDBFC3(_Buffer, i) /
                            (sqrt(getDBFC3_m(_Buffer, i)) + epsilon));

                        col.r -= ((_Train > 0.0 && !initTime) ? 1.0 : 0.0) * delta;
                    }
                    else if (lc == 23 && insideArea(txFC3s, px))
                    {
                        px -= txFC3s.xy;
                        uint i = px.y % 12;

                        float sum = 0.0;
                        for (uint j = 0; j < 128; j++) {
                            sum += getFC2a(_Buffer, j) * getWFC3(_Buffer, uint2(j, i));
                        }

                        sum += getBFC3(_Buffer, i);
                        col.r = sum;
                    }
                    else if (lc == 24 && insideArea(txFC3o, px))
                    {
                        px -= txFC3o.xy;
                        uint i = px.y % 12;
                        
                        float sum = 0.0;
                        for (uint j = 0; j < 12; j++) {
                            sum += exp(getFC3s(_Buffer, j));
                        }

                        col.r = exp(getFC3s(_Buffer, i)) / sum;
                    }
                }
                else if (insideArea(txB4Area, px))
                {
                    px -= txB4Area.xy;
                    [branch]
                    if (lc == 25 && insideArea(txDBFC3, px))
                    {
                        px -= txDBFC3.xy;
                        uint i = px.y % 12;
                        
                        col.r = getFC3o(_Buffer, i) - (i == _TargetClass ? 1.0 : 0.0);
                    }
                    else if (lc == 25 && insideArea(txDBFC3_m, px))
                    {
                        px -= txDBFC3_m.xy;
                        uint i = px.y % 12;

                        float dbFC3 = getFC3o(_Buffer, i) - (i == _TargetClass ? 1.0 : 0.0);
                        col.r = momentum(dbFC3, getDBFC3_m(_Buffer, i));
                    }
                    else if (lc == 26 && insideArea(txDWFC3, px))
                    {
                        px -= txDWFC3.xy;
                        uint i = px.x % 128;
                        uint j = px.y % 12;

                        col.r = getDBFC3(_Buffer, j) * getFC2a(_Buffer, i);
                    }
                    else if (lc == 26 && insideArea(txDWFC3_m, px))
                    {
                        px -= txDWFC3_m.xy;
                        uint i = px.x % 128;
                        uint j = px.y % 12;

                        float dwFC3 = getDBFC3(_Buffer, j) * getFC2a(_Buffer, i);
                        col.r = momentum(dwFC3, getDWFC3_m(_Buffer, i));
                    }
                    else if (lc == 27 && insideArea(txDBFC2, px))
                    {
                        px -= txDBFC2.xy;
                        uint i = px.y % 128;

                        float sum = 0.0;
                        for (uint j = 0; j < 12; j++) {
                            sum += getDBFC3(_Buffer, j) * getWFC3(_Buffer, uint2(i, j));
                        }

                        col.r = sum;
                    }
                    else if (lc == 27 && insideArea(txDBFC2_m, px))
                    {
                        px -= txDBFC2_m.xy;
                        uint i = px.y % 128;

                        float dbFC2 = 0.0;
                        for (uint j = 0; j < 12; j++) {
                            dbFC2 += getDBFC3(_Buffer, j) * getWFC3(_Buffer, uint2(i, j));
                        }

                        col.r = momentum(dbFC2, getDBFC2_m(_Buffer, i));
                    }
                    else if (lc == 28 && insideArea(txDWFC2, px))
                    {
                        px -= txDWFC2.xy;
                        uint i = px.x % 128;
                        uint j = px.y % 128;

                        col.r = getDBFC2(_Buffer, j) * dfn(getFC2s(_Buffer, i)) *
                            getFC1a(_Buffer, i);
                    }
                    else if (lc == 28 && insideArea(txDWFC2_m, px))
                    {
                        px -= txDWFC2_m.xy;
                        uint i = px.x % 128;
                        uint j = px.y % 128;

                        float dwFC2 = getDBFC2(_Buffer, j) * dfn(getFC2s(_Buffer, i)) *
                            getFC1a(_Buffer, i);
                        col.r = momentum(dwFC2, getDWFC2_m(_Buffer, i));
                    }
                    else if (lc == 29 && insideArea(txDBFC1, px))
                    {
                        px -= txDBFC1.xy;
                        uint i = px.y % 128;

                        float sum = 0.0;
                        for (uint j = 0; j < 128; j++) {
                            sum += getDBFC2(_Buffer, j) * dfn(getFC2s(_Buffer, i)) *
                                getWFC2(_Buffer, uint2(i, j));
                        }

                        col.r = sum;
                    }
                    else if (lc == 29 && insideArea(txDBFC1_m, px))
                    {
                        px -= txDBFC1_m.xy;
                        uint i = px.y % 128;

                        float dbFC1 = 0.0;
                        for (uint j = 0; j < 128; j++) {
                            dbFC1 += getDBFC2(_Buffer, j) * dfn(getFC2s(_Buffer, i)) *
                                getWFC2(_Buffer, uint2(i, j));
                        }

                        col.r = momentum(dbFC1, getDBFC1_m(_Buffer, i));
                    }
                    else if (lc == 30 && insideArea(txDWFC1, px))
                    {
                        px -= txDWFC1.xy;
                        uint i = px.x % 128;
                        uint j = px.y % 128;

                        col.r = getDBFC1(_Buffer, j) * dfn(getFC1s(_Buffer, i)) *
                            getL3Max(_Buffer, i);
                    }
                    else if (lc == 30 && insideArea(txDWFC1_m, px))
                    {
                        px -= txDWFC1_m.xy;
                        uint i = px.x % 128;
                        uint j = px.y % 128;

                        float dwFC1 = getDBFC1(_Buffer, j) * dfn(getFC1s(_Buffer, i)) *
                            getL3Max(_Buffer, i);
                        col.r = momentum(dwFC1, getDWFC1_m(_Buffer, uint2(i, j)));
                    }
                }
                else if (insideArea(txB3Area, px))
                {
                    px -= txB3Area.xy;
                    [branch]
                    if (lc == 31 && insideArea(txEMax3, px))
                    {
                        px -= txEMax3.xy;
                        uint i = px.y % 128;

                        float sum = 0.0;
                        for (uint j = 0; j < 128; j++) {
                            sum += getDBFC1(_Buffer, j) * dfn(getFC1s(_Buffer, i)) *
                                getWFC1(_Buffer, uint2(i, j));
                        }
                        col.r = sum;
                    }
                    else if (lc == 32 && insideArea(txEL3, px))
                    {
                        px -= txEL3.xy;
                        uint i = (px.x / 3) % 3;
                        uint j = px.x % 3;
                        uint k = px.y % 128;

                        col.r = getL3iMax(_Buffer, k) == int(i * 3 + j) ?
                            getEMax3(_Buffer, k) * dfn(getL3s(_Buffer, uint3(j, i, k))) :
                            0.0f;
                    }
                    else if (lc == 33 && insideArea(txDiL3, px))
                    {
                        px -= txDiL3.xy;
                        uint i = (px.x / 5) % 5;
                        uint j = px.x % 5;
                        uint k = px.y % 128;
                        uint i0 = i / 2;
                        uint j0 = j / 2;

                        col.r = ((i % 2 == 1) || (j % 2 == 1)) ?
                            0.0f : getEL3(_Buffer, uint3(i0, j0, k));
                    }
                    else if (lc == 34 && insideArea(txDbL3, px))
                    {
                        px -= txDbL3.xy;
                        uint i = px.y % 128;

                        col.r = getEMax3(_Buffer, i);
                    }
                    else if (lc == 34 && insideArea(txDbL3_m, px))
                    {
                        px -= txDbL3_m.xy;
                        uint i = px.y % 128;

                        float dbL3 = getEMax3(_Buffer, i);
                        col.r = momentum(dbL3, getDbL3_m(_Buffer, i));
                    }
                    else if (lc == 35 && insideArea(txDwL3, px))
                    {
                        px -= txDwL3.xy;
                        uint i = (px.x / 192) % 3;
                        uint j = (px.x / 64) % 3;
                        uint k = px.x % 64;
                        uint l = px.y % 128;

                        float sum = 0.0;
                        for (uint x = 0; x < 5; x++) {
                            for (uint y = 0; y < 5; y++) {
                                uint lx = x + i;
                                uint ly = y + j;
                                sum += getDiL3(_Buffer, uint3(x, y, l)) *
                                    getL2Max(_Buffer, uint3(lx, ly, k));
                            }
                        }
                        col.r = sum;
                    }
                    else if (lc == 35 && insideArea(txDwL3_m, px))
                    {
                        px -= txDwL3_m.xy;
                        uint i = (px.x / 192) % 3;
                        uint j = (px.x / 64) % 3;
                        uint k = px.x % 64;
                        uint l = px.y % 128;

                        float dwL3 = 0.0;
                        for (uint x = 0; x < 5; x++) {
                            for (uint y = 0; y < 5; y++) {
                                uint lx = x + i;
                                uint ly = y + j;
                                dwL3 += getDiL3(_Buffer, uint3(x, y, l)) *
                                    getL2Max(_Buffer, uint3(lx, ly, k));
                            }
                        }

                        col.r = momentum(dwL3, getDwL3_m(_Buffer, uint4(i, j, k, l)));
                    }
                }
                else if (insideArea(txB2Area, px))
                {
                    px -= txB2Area.xy;
                    [branch]
                    if (lc == 36 && insideArea(txPadL3, px))
                    {
                        px -= txPadL3.xy;
                        uint i = (px.x / 9) % 9;
                        uint j = px.x % 9;
                        uint k = px.y % 128;

                        col.r = (i < 2 || j < 2 || i > 6 || j > 6) ?
                            0.0 : getDiL3(_Buffer, uint3(i - 2, j - 2, k));
                    }
                    else if (lc == 37 && insideArea(txEL2Max, px))
                    {
                        px -= txEL2Max.xy;
                        uint i = (px.x / 7) % 7;
                        uint j = px.x % 7;
                        uint k = px.y % 64;

                        float sum = 0.0;
                        for (int l = 0; l < 128; l++) {
                            sum += getPadL3(_Buffer, uint3(i + 0, j + 0, l)) * getWL3(_Buffer, uint4(2, 2, k, l));
                            sum += getPadL3(_Buffer, uint3(i + 0, j + 1, l)) * getWL3(_Buffer, uint4(2, 1, k, l));
                            sum += getPadL3(_Buffer, uint3(i + 0, j + 2, l)) * getWL3(_Buffer, uint4(2, 0, k, l));
                            sum += getPadL3(_Buffer, uint3(i + 1, j + 0, l)) * getWL3(_Buffer, uint4(1, 2, k, l));
                            sum += getPadL3(_Buffer, uint3(i + 1, j + 1, l)) * getWL3(_Buffer, uint4(1, 1, k, l));
                            sum += getPadL3(_Buffer, uint3(i + 1, j + 2, l)) * getWL3(_Buffer, uint4(1, 0, k, l));
                            sum += getPadL3(_Buffer, uint3(i + 2, j + 0, l)) * getWL3(_Buffer, uint4(0, 2, k, l));
                            sum += getPadL3(_Buffer, uint3(i + 2, j + 1, l)) * getWL3(_Buffer, uint4(0, 1, k, l));
                            sum += getPadL3(_Buffer, uint3(i + 2, j + 2, l)) * getWL3(_Buffer, uint4(0, 0, k, l));
                        }
                        col.r = sum;
                    }
                    else if (lc == 38 && insideArea(txEL2, px))
                    {
                        px -= txEL2.xy;
                        uint i = (px.x / 14) % 14;
                        uint j = px.x % 14;
                        uint k = px.y % 64;
                        uint i0 = i / 2;
                        uint j0 = j / 2;

                        col.r = getL2iMax(_Buffer, uint3(i0, j0, k)) == int(i * 14 + j) ?
                            getEL2Max(_Buffer, uint3(i0, j0, k)) *
                            dfn(getL1s(_Buffer, uint3(j, i, k))) : 0.0f;
                    }
                    else if (lc == 39 && insideArea(txDbL2, px))
                    {
                        px -= txDbL2.xy;
                        uint i = px.y % 64;

                        float sum = 0.0;
                        for (uint x = 0; x < 7; x++) {
                            for (uint y = 0; y < 7; y++) {
                                sum += getEL2Max(_Buffer, uint3(x, y, i));
                            }
                        }
                        col.r = sum;
                    }
                    else if (lc == 39 && insideArea(txDbL2_m, px))
                    {
                        px -= txDbL2_m.xy;
                        uint i = px.y % 64;

                        float dbL2 = 0.0;
                        for (uint x = 0; x < 7; x++) {
                            for (uint y = 0; y < 7; y++) {
                                dbL2 += getEL2Max(_Buffer, uint3(x, y, i));
                            }
                        }

                        col.r = momentum(dbL2, getDbL2_m(_Buffer, i));
                    }
                    else if (lc == 40 && insideArea(txDwL2, px))
                    {
                        px -= txDwL2.xy;
                        uint i = (px.x / 96) % 3;
                        uint j = (px.x / 32) % 3;
                        uint k = px.x % 32;
                        uint l = px.y % 64;

                        float sum = 0.0;
                        for (uint x = 0; x < 14; x++) {
                            for (uint y = 0; y < 14; y++) {
                                uint lx = x + i;
                                uint ly = y + j;
                                sum += getEL2(_Buffer, uint3(x, y, l)) *
                                    getL1Max(_Buffer, uint3(lx, ly, k));
                            }
                        }
                        col.r = sum;
                    }
                    else if (lc == 40 && insideArea(txDwL2_m, px))
                    {
                        px -= txDwL2_m.xy;
                        uint i = (px.x / 96) % 3;
                        uint j = (px.x / 32) % 3;
                        uint k = px.x % 32;
                        uint l = px.y % 64;

                        float dwL2 = 0.0;
                        for (uint x = 0; x < 14; x++) {
                            for (uint y = 0; y < 14; y++) {
                                uint lx = x + i;
                                uint ly = y + j;
                                dwL2 += getEL2(_Buffer, uint3(x, y, l)) *
                                    getL1Max(_Buffer, uint3(lx, ly, k));
                            }
                        }

                        col.r = momentum(dwL2, getDwL2_m(_Buffer, uint4(i, j, k, l)));
                    }
                }
                else if (insideArea(txB1Area, px))
                {
                    px -= txB1Area.xy;
                    [branch]
                    if (lc == 41 && insideArea(txPadL2, px))
                    {
                        px -= txPadL2.xy;
                        uint i = (px.x / 18) % 18;
                        uint j = px.x % 18;
                        uint k = px.y % 64;

                        col.r = (i < 2 || j < 2 || i > 15 || j > 15) ?
                            0.0f : getEL2(_Buffer, uint3(i - 2, j - 2, k));
                    }
                    else if (lc == 42 && insideArea(txEL1Max, px))
                    {
                        px -= txEL1Max.xy;
                        uint i = (px.x / 16) % 16;
                        uint j = px.x % 16;
                        uint k = px.y % 32;

                        float sum = 0.0;
                        for (int l = 0; l < 64; l++) {
                            sum += getPadL2(_Buffer, uint3(i + 0, j + 0, l)) * getWL2(_Buffer, uint4(2, 2, k, l));
                            sum += getPadL2(_Buffer, uint3(i + 0, j + 1, l)) * getWL2(_Buffer, uint4(2, 1, k, l));
                            sum += getPadL2(_Buffer, uint3(i + 0, j + 2, l)) * getWL2(_Buffer, uint4(2, 0, k, l));
                            sum += getPadL2(_Buffer, uint3(i + 1, j + 0, l)) * getWL2(_Buffer, uint4(1, 2, k, l));
                            sum += getPadL2(_Buffer, uint3(i + 1, j + 1, l)) * getWL2(_Buffer, uint4(1, 1, k, l));
                            sum += getPadL2(_Buffer, uint3(i + 1, j + 2, l)) * getWL2(_Buffer, uint4(1, 0, k, l));
                            sum += getPadL2(_Buffer, uint3(i + 2, j + 0, l)) * getWL2(_Buffer, uint4(0, 2, k, l));
                            sum += getPadL2(_Buffer, uint3(i + 2, j + 1, l)) * getWL2(_Buffer, uint4(0, 1, k, l));
                            sum += getPadL2(_Buffer, uint3(i + 2, j + 2, l)) * getWL2(_Buffer, uint4(0, 0, k, l));
                        }
                        col.r = sum;
                    }
                    else if (lc == 43 && insideArea(txEL1, px))
                    {
                        px -= txEL1.xy;
                        uint i = (px.x / 32) % 32;
                        uint j = px.x % 32;
                        uint k = px.y % 32;
                        uint i0 = i / 2;
                        uint j0 = j / 2;

                        col.r = getL1iMax(_Buffer, uint3(i0, j0, k)) == int(i * 32 + j) ?
                            getEL1Max(_Buffer, uint3(i0, j0, k)) *
                            dfn(getL1s(_Buffer, uint3(j, i, k))) : 0.0f;
                    }
                    else if (lc == 44 && insideArea(txDiL1, px))
                    {
                        px -= txDiL1.xy;
                        uint i = px.y % 63;
                        uint j = px.x % 63;
                        uint k = (px.x / 63) % 8 + (px.y / 63) * 8;
                        uint i0 = i / 2;
                        uint j0 = j / 2;

                        col.r = ((i % 2 == 1) || (j % 2 == 1)) ?
                            0.0f : getEL1(_Buffer, uint3(i0, j0, k));
                    }
                    else if (lc == 45 && insideArea(txDbL1, px))
                    {
                        px -= txDbL1.xy;
                        uint i = px.y % 32;

                        float sum = 0.0;
                        for (uint x = 0; x < 16; x++) {
                            for (uint y = 0; y < 16; y++) {
                                sum += getEL1Max(_Buffer, uint3(x, y, i));
                            }
                        }
                        col.r = sum;
                    }
                    else if (lc == 45 && insideArea(txDbL1_m, px))
                    {
                        px -= txDbL1_m.xy;
                        uint i = px.y % 32;
                        float dbL1 = 0.0;
                        for (uint x = 0; x < 16; x++) {
                            for (uint y = 0; y < 16; y++) {
                                dbL1 += getEL1Max(_Buffer, uint3(x, y, i));
                            }
                        }
                        col.r = momentum(dbL1, getDbL1_m(_Buffer, i));
                    }
                    else if (lc == 46 && insideArea(txDwL1, px))
                    {
                        px -= txDwL1.xy;
                        uint i = (px.x / 9) % 3;
                        uint j = (px.x / 3) % 3;
                        uint k = px.x % 3;
                        uint l = px.y % 32;

                        float sum = 0.0f;
                        for (uint x = 0; x < 63; x++) {
                            for (uint y = 0; y < 63; y++) {
                                uint lx = x + i;
                                uint ly = y + j;
                                // sum += getDiL1(_Buffer, uint3(x, y, l)) *
                                //     testImage(lx, ly, k);

                                sum += getDiL1(_Buffer, uint3(x, y, l)) *
                                    (_CamIn.Load(int3(ly, 64 - lx, 0))[k]);
                            }
                        }

                        col.r = sum;
                    }
                    else if (lc == 47 && insideArea(txDwL1_m, px))
                    {
                        px -= txDwL1_m.xy;
                        uint i = (px.x / 9) % 3;
                        uint j = (px.x / 3) % 3;
                        uint k = px.x % 3;
                        uint l = px.y % 32;

                        float dwL1 = 0.0f;
                        for (uint x = 0; x < 63; x++) {
                            for (uint y = 0; y < 63; y++) {
                                uint lx = x + i;
                                uint ly = y + j;
                                dwL1 += getDiL1(_Buffer, uint3(x, y, l)) *
                                    testImage(lx, ly, k);
                            }
                        }

                        col.r = momentum(dwL1, getDwL1_m(_Buffer, uint4(i, j, k, l)));
                    }
                }

                lc = (lc + 1) % (_Train > 0 ? LAYERS_TRAIN : LAYERS_CLASSIFY);
                StoreValue(txLC, lc, col, px);
                return col;
            }
            ENDCG
        }
    }
}
