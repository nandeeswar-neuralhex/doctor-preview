import re

def optimize_file():
    with open('/Users/nandeeswar/Desktop/Doctor-preview-main/runpod_v2/src/face_swapper.py', 'r') as f:
        content = f.read()
    
    # Replace the slow inside _swap_single_face
    new_swap = """
        try:
            res = self.swapper.get(frame, source_face, target_face, paste_back=False)
            if res is None:
                return frame
            bgr_fake, M = res
            return self._paste_back(
                frame, 
                bgr_fake, 
                source_face.kps, 
                source_face.bbox,
                source_face=source_face, 
                session_id=session_id
            )
        except Exception as e:
            print(f"Swap error: {e}")
            return frame
"""
    content = re.sub(
        r"return self\.swapper\.get\(frame, source_face, target_face, paste_back=True\)\n.*?except Exception as e:",
        new_swap.strip() + "\n        except Exception as e:", 
        content, flags=re.DOTALL)
        
    with open('/Users/nandeeswar/Desktop/Doctor-preview-main/runpod_v2/src/face_swapper.py', 'w') as f:
        f.write(content)

optimize_file()
