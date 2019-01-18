from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import Mat4 #GeomVertexFormat, GeomVertexWriter,  
import math
world_pts = [
        (3.313, 4.824),
        (1.337, 5.325),
        (0, 0),
        (1.936, -0.501),
        (13.290, 5.325),
        (11.314, 4.824),
        (12.691, -0.501),
        (14.627, 0),
]



color = (128, 255, 128)
aa = False
class MyApp(ShowBase):
 
    def __init__(self):
        ShowBase.__init__(self)
        #self.scene = self.loader.loadModel("models/environment")
        #self.disableMouse()
        self.scene = self.loader.loadModel("Target.egg")
        self.scene.reparentTo(self.render)
        self.scene.setPos(0, 0, -.254)
        self.scene.setHpr(0, 90, 0)
        self.taskMgr.add(self.cameraTask, "CameraTask")
        #self.taskMgr.add(self.spinCameraTask, "sct")
    def spinCameraTask(self, task):
        angleDegrees = task.time * 6.0
        angleRadians = angleDegrees * (math.pi / 180.0)
        self.camera.setPos(20 * math.sin(angleRadians), -20.0 * math.cos(angleRadians) + 20, 50)
        self.camera.setHpr(angleDegrees, -90, 0)
        return Task.cont

    def cameraTask(self, taskinfo):
        global aa
        if True:#not aa:
            self.camera.setPos(0, -50, 0)#20 * 2.54, 10 * 2.54, 40 * 2.54)
            self.camera.setHpr(0, math.sin(taskinfo.time) * 5, 0)
            aa = True
            m = Mat4(self.camera.getMat())
            m.invertInPlace()
            self.mouseInterfaceNode.setMat(m)
        print(self.camera.getPos(), self.camera.getHpr())
        return Task.cont
 

if __name__ == '__main__':
    app = MyApp()
    #app.camera.disableMouse()
    app.run()

