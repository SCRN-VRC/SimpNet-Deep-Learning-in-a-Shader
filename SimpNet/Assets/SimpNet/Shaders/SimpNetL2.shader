Shader "SimpNet/SimpNetL2"
{
    Properties
    {
        _Layer1 ("Layer 1", 2D) = "black" {}
        _L2Gradients ("Layer 2 Gradients", 2D) = "black" {}
        _FrameBuffer ("Layer 2 Buffer", 2D) = "black" {}
    }
    SubShader
    {
        
        Pass
        {
            Name "SimpNet Layer 2"

            CGPROGRAM
            #include "UnityCustomRenderTexture.cginc"
            #include "Includes/SimpNetLayout.cginc"
            #include "Includes/SimpNetFuncs.cginc"
            #pragma vertex CustomRenderTextureVertexShader
            #pragma fragment pixel_shader
            #pragma target 5.0

            //RWStructuredBuffer<float4> buffer : register(u1);
            Texture2D<float3> _Layer1;
            Texture2D<float3> _L2Gradients;
            Texture2D<float3> _FrameBuffer;
            float4 _Layer1_TexelSize;
            float4 _L2Gradients_TexelSize;
            float4 _FrameBuffer_TexelSize;

            float3 pixel_shader (v2f_customrendertexture IN) : SV_TARGET
            {
                int2 px = _FrameBuffer_TexelSize.zw * IN.globalTexcoord.xy;
                float3 col = _FrameBuffer.Load(int3(px, 0));

                [branch]
                if (insideArea(txKern2Area, px))
                {
                    if (_Time.y <= 1.0)
                    {
                        // col.r = px.y * _FrameBuffer_TexelSize.z + px.x;
                        // col.r = rand(col.r) * 0.003472;
                        
                        // Debugging
                        px -= txKern2Area.xy;
                        int i = px.y % 3;
                        int j = px.x % 3;
                        int k = (px.y / 3);
                        int l = (px.x / 3);
                        col.r = (i + j + k + l) / 1000.0;
                    }
                }
                else if (insideArea(txBias2Area, px))
                {
                    if (_Time.y <= 1.0)
                    {
                        // col.r = px.y * _FrameBuffer_TexelSize.z + px.x;
                        // col.r = rand(col.r) * 0.5;

                        // Debugging
                        px -= txBias2Area.xy;
                        col.r = 1.0 - (px.y / 64.0) - 0.5;
                    }
                }
                else if (insideArea(txConv2Area, px))
                {
                    px -= txConv2Area.xy;
                    int i = px.y % 14;
                    int j = px.x % 14;
                    int k = px.x / 14 + (px.y / 14) * 8;
                    int i0 = i, i1 = i + 1, i2 = i + 2;
                    int j0 = j, j1 = j + 1, j2 = j + 2;

                    float sum = 0.0;
                    for (int l = 0; l < 32; l++) {
                        sum += getMax1(_Layer1, int3(j0, i0, l)) * getKern2(_FrameBuffer, int4(0, 0, l, k));
                        sum += getMax1(_Layer1, int3(j0, i1, l)) * getKern2(_FrameBuffer, int4(0, 1, l, k));
                        sum += getMax1(_Layer1, int3(j0, i2, l)) * getKern2(_FrameBuffer, int4(0, 2, l, k));
                        sum += getMax1(_Layer1, int3(j1, i0, l)) * getKern2(_FrameBuffer, int4(1, 0, l, k));
                        sum += getMax1(_Layer1, int3(j1, i1, l)) * getKern2(_FrameBuffer, int4(1, 1, l, k));
                        sum += getMax1(_Layer1, int3(j1, i2, l)) * getKern2(_FrameBuffer, int4(1, 2, l, k));
                        sum += getMax1(_Layer1, int3(j2, i0, l)) * getKern2(_FrameBuffer, int4(2, 0, l, k));
                        sum += getMax1(_Layer1, int3(j2, i1, l)) * getKern2(_FrameBuffer, int4(2, 1, l, k));
                        sum += getMax1(_Layer1, int3(j2, i2, l)) * getKern2(_FrameBuffer, int4(2, 2, l, k));
                    }

                    sum += _FrameBuffer.Load(int3(txBias2Area.xy + int2(0, k), 0)).x;
                    col.r = actFn(sum);
                }
                else if (insideArea(txMax2Area, px))
                {
                    px -= txMax2Area.xy;
                    int i = px.y % 7;
                    int j = px.x % 7;
                    int k = px.x / 7 + (px.y / 7) * 8;
                    int i0 = i * 2, i1 = i0 + 1;
                    int j0 = j * 2, j1 = j0 + 1;

                    float m = getConv2(_FrameBuffer, int3(j0, i0, k));
                    m = max(m, getConv2(_FrameBuffer, int3(j0, i1, k)));
                    m = max(m, getConv2(_FrameBuffer, int3(j1, i0, k)));
                    m = max(m, getConv2(_FrameBuffer, int3(j1, i1, k)));
                    col.r = m;
                }
                else if (insideArea(txiMax2Area, px))
                {
                    px -= txiMax2Area.xy;
                    int i = px.y % 7;
                    int j = px.x % 7;
                    int k = px.x / 7 + (px.y / 7) * 8;
                    int i0 = i * 2, i1 = i0 + 1;
                    int j0 = j * 2, j1 = j0 + 1;

                    float bu;
                    float m = getConv2(_FrameBuffer, int3(j0, i0, k));
                    col.r = i0 * 14 + j0;
                    
                    m = max(m, bu = getConv2(_FrameBuffer, int3(j0, i1, k)));
                    col.r = (m == bu) ? (i1 * 14 + j0) : col.r;
                    
                    m = max(m, bu = getConv2(_FrameBuffer, int3(j1, i0, k)));
                    col.r = (m == bu) ? (i0 * 14 + j1) : col.r;
                    
                    m = max(m, bu = getConv2(_FrameBuffer, int3(j1, i1, k)));
                    col.r = (m == bu) ? (i1 * 14 + j1) : col.r;
                }
                return col;
            }
            ENDCG
        }
    }
}