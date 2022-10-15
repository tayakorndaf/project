from flask import Flask,render_template,request,Response
from camera import Squatt
from camera import leglungess
from camera import toysoldierr
from camera import Highkneeruns
from camera import standingsidecrunch
from camera import jumpslap
from camera import Video


app = Flask(__name__)


c=0
def gen(camera):

    while True:
        frame,c=camera.get_frame()
        
        yield(b' --frame\r\n'
              b'Content-Type:  image/jpeg\r\n\r\n'+ frame +
              b'\r\n\r\n') 
        print(c)
        

 
    
        





@app.route('/squatt')
def squa():
    return Response(gen(Squatt()),
    mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app.route('/leglungess')
def leglunge():
    return Response(gen(leglungess()),
    mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app.route('/toysoldierr')
def toysoldie():
    return Response(gen(toysoldierr()),
    mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app.route('/Highkneerunss')
def Highkneeru():
    return Response(gen(Highkneeruns()),
    mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app.route('/standingsidecrunchh')
def standingsidecrun():
    return Response(gen(standingsidecrunch()),
    mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app.route('/jumpslapp')
def jumpsl():
    return Response(gen(jumpslap()),
    mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app.route('/video')
def video():
    return Response(gen(Video()),
    mimetype='multipart/x-mixed-replace; boundary=frame')
    
    
    
    
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/insertdata')
def insert():
    return render_template("insertdata.html")

@app.route('/squat')
def sq():
    return render_template("t/squat.html")

@app.route('/leglunges')
def ll():
    return render_template("t/leglunges.html")

@app.route('/toysoldier')
def ts():
    return render_template("t/toysoldier.html")

@app.route('/highkneeruns')
def hk():
    return render_template("t/highkneeruns.html")

@app.route('/jumpslap')
def js():
    return render_template("t/jumpslap.html")

@app.route('/standingsidecrunch')
def sc():
    return render_template("t/standingsidecrunch.html")



@app.route('/startSQ')
def ssq():
    
    return render_template("t/startSQ.html")

@app.route('/startLL')
def sll():
    return render_template("t/startLL.html")
    
@app.route('/startTS')
def sts():
    return render_template("t/startTS.html")

@app.route('/startSC')
def ssc():
    return render_template("t/startSC.html")

@app.route('/startHK')
def shk():
    return render_template("t/startHK.html")

@app.route('/startJS')
def sjs():
    return render_template("t/startJS.html")



@app.route('/scoresq')
def scoresq():
    
    return render_template("end/scoresq.html",counter_Squat=c)


@app.route('/scorehk')
def scorehk():
    
    return render_template("end/scorehk.html",counter_Highkneeruns=c)


@app.route('/scorejs')
def scorejs():
    
    return render_template("end/scorejs.html",counter_Jumpslap=c)


@app.route('/scorell')
def scorell():
    return render_template("end/scorell.html",counter_leglungess=c)


@app.route('/scoresc')
def scoresc():
    return render_template("end/scoresc.html",counter_standingsidecrunch=c)


@app.route('/scorets')
def scorets():
    return render_template("end/scorets.html",counter_toysoldier=c)



@app.route('/menu')
def menu():
    myname = request.args.get('name')
    Value = request.args.get('value')
    return render_template("menu.html",myname= myname,value=Value)



if __name__ == "__main__":
    app.run(debug=True)