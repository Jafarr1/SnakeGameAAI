import base64
import imageio
import IPython


class EnvVideoRecorder:
    """
    Record Gym environment frames, save video, and embed in Jupyter.
    Henrik Strøm, November 2022
    """
    
    def __init__(self, env):
        self._frame_buffer = []
        self._rewards = []
        self._env = env
        self._saved = False
        
    def __len__(self):
        return len(self._frame_buffer)
    
    def rewards(self):
        return self._rewards
        
    def reset(self):
        self._frame_buffer = []
        self._rewards = []
        return self._env.reset()
        
    def render(self):
        frame = self._env.render()
        self._frame_buffer.append(frame)
        return frame
        
    def step(self, action):
        observation, reward, terminated, truncated, info = self._env.step(action)
        self._rewards.append(reward)
        return observation, reward, terminated, truncated, info
    
    def save(self, filename, fps=30):
        self._filename = filename
        with imageio.get_writer(filename, fps=fps) as flick:
            for frame in self._frame_buffer:
                flick.append_data(frame)
        self._saved = True
        
    def embed_jupyter(self):
        if self._saved:
            flick = open(self._filename, 'rb')
            video = flick.read()
            flick.close()
            base64_video = base64.b64encode(video)
            html_tag = '''
            <video with="608" height="400" controls>
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                Please use a browser that supports the HTML video tag.
            </video>
            '''.format(base64_video.decode())
        return IPython.display.HTML(html_tag)