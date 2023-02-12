
import functools
from typing import Iterable
import pyglet

import guis.observable as observable

class GuiData(observable.Observable):
    __slots__ = ("__desired_view_state",
                 "__desired_window_width",
                 "__desired_window_height")
    
    def __init__(self) -> None:
        super().__init__()
        self.__desired_view_state: str = ""
    
    @property
    def desired_view_state(self) -> str:
        return self.__desired_view_state
    
    @desired_view_state.setter
    @observable.notifies_observers
    def desired_view_state(self, desired_view_state: str) -> None:
        self.__desired_view_state = desired_view_state
    
    @property
    def desired_window_width(self) -> int:
        return self.__desired_window_width
    
    @desired_window_width.setter
    @observable.notifies_observers
    def desired_window_width(self, desired_window_width: int) -> None:
        self.__desired_window_width = desired_window_width
    
    @property
    def desired_window_height(self) -> int:
        return self.__desired_window_height
    
    @desired_window_height.setter
    @observable.notifies_observers
    def desired_window_height(self, desired_window_height: int) -> None:
        self.__desired_window_height = desired_window_height

class Gui(pyglet.window.Window, observable.Observer):
    
    def __init__(self, gui_data: "GuiData", view_states: dict[str, pyglet.gui.widgets.WidgetBase]) -> None:
        super().__init__(width=100, height=100, caption="Hello")
        observable.Observer.__init__(self)
        self.__data: GuiData = gui_data
        self.__data.assign_observers(self)
        
        self.__view_states: dict[str, pyglet.gui.widgets.WidgetBase] = view_states
        self.__current_view_state: str | None = None
        self.__data.desired_view_state = None

        self.__window = pyglet.window.Window(width=100, height=100, caption="Hello")
        # self.__frame = pyglet.gui.Frame(self.__window)

        self.__batch = pyglet.graphics.Batch()
        # self.__group = pyglet.graphics.Group()
        
        self.switch_to()
        self.set_visible()
        self.__data.notify_all()
    
    def run(self) -> None:
        pyglet.app.run()
    
    def on_draw(self) -> None:
        self.clear()
        self.__batch.draw()
    
    @property
    def batch(self) -> pyglet.graphics.Batch:
        return self.__batch
    
    @property
    def current_view_state(self) -> str:
        return self.__current_view_state
    
    def update_observer(self, observable: observable.Observable) -> None:
        desired_view_state: str | None = self.__data.desired_view_state
        if self.__current_view_state != desired_view_state:
            self.__data.clear_observers()
            self.__data.assign_observers(self)
            if desired_view_state is not None:
                self.__data.assign_observers(self.__view_states[desired_view_state])
            self.__current_view_state = desired_view_state
            self.__data.notify_all()

class GuiView(pyglet.text.Label, observable.Observer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        observable.Observer.__init__(self)
    
    def update_observer(self, observable: observable.Observable) -> None:
        return None

if __name__ == "__main__":
    gui_data = GuiData()
    gui = Gui(gui_data, {1: GuiView("Hello World", font_name="Times New Roman", font_size=36, x=10, y=10, anchor_x="left", anchor_y="bottom")})
    gui.run()
