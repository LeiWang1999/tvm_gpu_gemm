#[version = "0.0.5"]
@main = primfn(A_1: handle, W_1: handle, Conv_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float16), float16, [16, 14, 14, 16, 16, 16], []),
             W: Buffer(W_2: Pointer(float16), float16, [3, 3, 16, 32, 16, 16], []),
             Conv: Buffer(Conv_2: Pointer(float32), float32, [16, 14, 14, 32, 16, 16], [])}
  buffer_map = {A_1: A, W_1: W, Conv_1: Conv} {
  for (n: int32, 0, 16) {
    for (h: int32, 0, 14) {
      for (w: int32, 0, 14) {
        for (o: int32, 0, 32) {
          for (nn: int32, 0, 16) {
            for (oo: int32, 0, 16) {
              Conv_3: Buffer(Conv_2, float32, [25690112], [])[((((((n*1605632) + (h*114688)) + (w*8192)) + (o*256)) + (nn*16)) + oo)] = 0f32
              for (ic: int32, 0, 16) {
                for (kh: int32, 0, 3) {
                  for (kw: int32, 0, 3) {
                    for (ii: int32, 0, 16) {
                      let cse_var_5: int32 = (o*256)
                      let cse_var_4: int32 = (nn*16)
                      let cse_var_3: int32 = (h + kh)
                      let cse_var_2: int32 = (w + kw)
                      let cse_var_1: int32 = ((((((n*1605632) + (h*114688)) + (w*8192)) + cse_var_5) + cse_var_4) + oo)
                      Conv_3[cse_var_1] = (Conv_3[cse_var_1] + (cast(float32, @tir.if_then_else(((((1 <= cse_var_3) && (cse_var_3 < 15)) && (1 <= cse_var_2)) && (cse_var_2 < 15)), A_3: Buffer(A_2, float16, [12845056], [])[(((((((((n*802816) + (h*57344)) + (kh*57344)) + (w*4096)) + (kw*4096)) + (ic*256)) + cse_var_4) + ii) - 61440)], 0f16, dtype=float16))*cast(float32, W_3: Buffer(W_2, float16, [1179648], [])[((((((kh*393216) + (kw*131072)) + (ic*8192)) + cse_var_5) + (ii*16)) + oo)])))
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

#[metadata]
{
  "root": 1, 
  "nodes": [
    {
      "type_key": ""
    }, 
    {
      "type_key": "Map", 
      "keys": [
        "IntImm"
      ], 
      "data": [2]
    }, 
    {
      "type_key": "Array", 
      "data": [3, 4]
    }, 
    {
      "type_key": "IntImm", 
      "attrs": {
        "dtype": "bool", 
        "span": "0", 
        "value": "1"
      }
    }, 
    {
      "type_key": "IntImm", 
      "attrs": {
        "dtype": "bool", 
        "span": "0", 
        "value": "1"
      }
    }
  ], 
  "b64ndarrays": [], 
  "attrs": {"tvm_version": "0.11.dev0"}
}