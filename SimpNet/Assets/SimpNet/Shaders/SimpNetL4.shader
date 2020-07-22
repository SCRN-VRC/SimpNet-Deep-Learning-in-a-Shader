Shader "SimpNet/SimpNetL4"
{
    Properties
    {
        _Layer3 ("Layer 3", 2D) = "black" {}
        _L4Gradients ("Layer 4 Gradients", 2D) = "black" {}
        _FrameBuffer ("Layer 4 Buffer", 2D) = "black" {}
    }
    SubShader
    {
        
        Pass
        {
            Name "SimpNet Layer 4"

            CGPROGRAM
            #include "UnityCustomRenderTexture.cginc"
            #include "Includes/SimpNetLayout.cginc"
            #include "Includes/SimpNetFuncs.cginc"
            #pragma vertex CustomRenderTextureVertexShader
            #pragma fragment pixel_shader
            #pragma target 5.0

            //RWStructuredBuffer<float4> buffer : register(u1);
            Texture2D<float3> _Layer3;
            Texture2D<float3> _L4Gradients;
            Texture2D<float3> _FrameBuffer;
            float4 _Layer3_TexelSize;
            float4 _L4Gradients_TexelSize;
            float4 _FrameBuffer_TexelSize;

            float3 pixel_shader (v2f_customrendertexture IN) : SV_TARGET
            {
                int2 px = _FrameBuffer_TexelSize.zw * IN.globalTexcoord.xy;
                float3 col = _FrameBuffer.Load(int3(px, 0));

                [branch]
                if (insideArea(txW1Area, px))
                {
                    if (_Time.y <= 1.0)
                    {
                        // col.r = px.y * _FrameBuffer_TexelSize.z + px.x;
                        // col.r = rand(col.r) * 0.001953125;
                        
                        // Debugging
                        px -= txW1Area.xy;
                        int i = px.y % 2;
                        int j = px.x % 2;
                        int k = (px.y / 2);
                        int l = (px.x / 2);
                        col.r = (i * (j + i)) * k / float(l * k + 1);
                    }
                }
                else if (insideArea(txW1BiasArea, px))
                {
                    if (_Time.y <= 1.0)
                    {
                        // col.r = px.y * _FrameBuffer_TexelSize.z + px.x;
                        // col.r = rand(col.r) * 0.5;

                        // Debugging
                        px -= txW1BiasArea.xy;
                        col.r = (px.y % 8) / 8.0;
                    }
                }
                else if (insideArea(txFC1s, px))
                {
                    px -= txFC1s.xy;
                    int i = px.y;

                    float sum = 0.0;
                    for (int k = 0; k < 2; k++) {
                        for (int l = 0; l < 2; l++) {
                            for (int j = 0; j < 128; j++) {
                                sum += getMax3(_Layer3, int3(l, k, j)) * getW1(_FrameBuffer, int4(k, l, j, i));
                            }
                        }
                    }
                    sum += _FrameBuffer.Load(int3(txW1BiasArea.xy + int2(0, i), 0)).x;
                    col.r = sum;
                }
                else if (insideArea(txFC1a, px))
                {
                    px -= txFC1a.xy;
                    col.r = actFn(_FrameBuffer.Load(int3(txFC1s.xy + int2(0, px.y), 0)).x);
                }

                return col;
            }
            ENDCG
        }
    }
}