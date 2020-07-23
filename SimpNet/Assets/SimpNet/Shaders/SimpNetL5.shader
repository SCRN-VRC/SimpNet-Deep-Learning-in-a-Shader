Shader "SimpNet/SimpNetL5"
{
    Properties
    {
        _Layer4 ("Layer 4", 2D) = "black" {}
        _L5Gradients ("Layer 5 Gradients", 2D) = "black" {}
        _FrameBuffer ("Layer 5 Buffer", 2D) = "black" {}
    }
    SubShader
    {
        
        Pass
        {
            Name "SimpNet Layer 5"

            CGPROGRAM
            #include "UnityCustomRenderTexture.cginc"
            #include "Includes/SimpNetLayout.cginc"
            #include "Includes/SimpNetFuncs.cginc"
            #pragma vertex CustomRenderTextureVertexShader
            #pragma fragment pixel_shader
            #pragma target 5.0

            //RWStructuredBuffer<float4> buffer : register(u1);
            Texture2D<float3> _Layer4;
            Texture2D<float3> _L5Gradients;
            Texture2D<float3> _FrameBuffer;
            float4 _Layer4_TexelSize;
            float4 _L5Gradients_TexelSize;
            float4 _FrameBuffer_TexelSize;

            float3 pixel_shader (v2f_customrendertexture IN) : SV_TARGET
            {
                int2 px = _FrameBuffer_TexelSize.zw * IN.globalTexcoord.xy;
                float3 col = _FrameBuffer.Load(int3(px, 0));
                int ct = int(_FrameBuffer.Load(int3(_FrameBuffer_TexelSize.zw - 1, 0)).x);

                [branch]
                if (ct == 0 && insideArea(txW2Area, px))
                {
                    if (_Time.y <= 1.0)
                    {
                        // col.r = px.y * _FrameBuffer_TexelSize.z + px.x;
                        // col.r = rand(col.r) * 0.0078125;
                        
                        // Debugging
                        px -= txW2Area.xy;
                        int i = px.y;
                        int j = px.x;
                        col.r = (i + j) / (128.0 * 128.0);
                    }
                }
                else if (ct == 0 && insideArea(txW2BiasArea, px))
                {
                    if (_Time.y <= 1.0)
                    {
                        // col.r = px.y * _FrameBuffer_TexelSize.z + px.x;
                        // col.r = rand(col.r) * 0.5;

                        // Debugging
                        px -= txW2BiasArea.xy;
                        col.r = 1.0 / (px.y + 1.0);
                    }
                }
                else if (ct == 1 && insideArea(txFC2s, px))
                {
                    px -= txFC2s.xy;
                    int i = px.y;

                    float sum = 0.0;
                    for (int j = 0; j < 128; j++) {
                        sum += _Layer4.Load(int3(txFC1a.xy + int2(0, j), 0)).x *
                            _FrameBuffer.Load(int3(txW2Area.xy + int2(i, j), 0)).x;
                    }
                    sum += _FrameBuffer.Load(int3(txW2BiasArea.xy + int2(0, i), 0)).x;
                    col.r = sum;
                }
                else if (ct == 2 && insideArea(txFC2a, px))
                {
                    px -= txFC2a.xy;
                    col.r = actFn(_FrameBuffer.Load(int3(txFC2s.xy + int2(0, px.y), 0)).x);
                }

                ct = min(ct + 1, 3);
                StoreValue(_FrameBuffer_TexelSize.zw - 1, ct, col.r, px);
                return col;
            }
            ENDCG
        }
    }
}