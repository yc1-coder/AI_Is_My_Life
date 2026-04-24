from datetime import datetime
import json
class Score:
    """成绩类-拆解：独立管理成绩"""
    def __init__(self,subject,score):
        self.subject = subject
        self.score = self._validate_score(score)
        self.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S") #时间戳
    def _validate_score(self,score):
        """验证成绩有效性"""
        if not 0 <= score <=100:
            raise ValueError(f"成绩{score}不在有效范围（0～100）")
        return score
    def update_score(self,new_score):
        """更新成绩"""
        self.score = self._validate_score(new_score)
        self.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"{self.subject}成绩已经更新为{self.score}分"
    def get_grade(self):
        """获取等级"""
        if self.score >=90:
            return "A"
        elif self.score >=80:
            return "B"
        elif self.score >=70:
            return "C"
        elif self.score >=60:
            return "D"
        else:
            return "F"
    def to_dict(self):
        return{
            "subject":self.subject,
            "score":self.score,
            "grade":self.get_grade(),
            "update_time":self.update_time
        }
class Student:
    """学生类 - 重构：优化结构和职责"""
    def __init__(self,name,age,sex,scores=None):
        self.name = name
        self.age = self._validate_age(age)
        self.sex = self._validate_sex(sex)
        self.scores ={}    #多课目成绩管理
        if scores:
            for subject,score in scores.items():
                self.add_score(subject,score)
        self.create_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    def _validate_age(self,age):
        """验证年龄有效性"""
        if not 5 <= age <= 28:
            raise ValueError(f"年龄{age}不在有效范围（5～28岁）之间")
        return age
    def _validate_sex(self,sex):
        """验证性别有效性"""
        if sex not in ["男","女"]:
            raise ValueError(f"性别{sex}必须是男或女")
        return sex
    def add_score(self,subject,score):
        """添加科目成绩"""
        self.scores[subject] = Score(subject,score)
        return f"已添加{subject}成绩:{score}分"
    def update_score(self,subject,new_score):
        """更新指定科目成绩"""
        if subject not in self.scores:
            raise KeyError(f"科目{subject}不存在")
        return self.scores[subject].update_score(new_score)
    def get_average_score(self):
        """计算平均分"""
        if not self.scores:
            return 0
        # total = sum(s.score for s in self.scores.values())
        total = 0
        for subject_name,score_object in self.scores.items():
            total += score_object.score
        return round(total / len(self.scores),2)
    def get_best_subject(self):
        """获取最高分科目"""
        if not self.scores:
            return None
        best = max(self.scores.items(),key=lambda x:x[1].score)
        return best[0],best[1].score
    def introduce(self):
        """自我介绍"""
        avg_score = self.get_average_score()
        intro = f"大家好，我叫{self.name},今年{self.age}岁了，{self.sex}生"
        if self.scores:
            intro += f"\n共有{len(self.scores)}门科目成绩"
            intro += f"\n平均成绩:{avg_score}分"
            best_subject,best_score = self.get_best_subject()
            intro += f"\n最高分:{best_subject},{best_score}分"
        return intro
    def get_all_scores(self):
        """获取成绩详情"""
        if not self.scores:
            return "暂无成绩记录"
        result = f"\n{'='*40}\n{self.name}的成绩单\n{'='*40}"
        for subject,score_obj in self.scores.items():
            result += f"\n{subject}:{score_obj.score}分(等级：{score_obj.get_grade()})"
        result += f"\n{'='*40}"
        result += f"\n平均分:{self.get_average_score()}分"
        return result
    def to_dict(self):
        """专为字典(用于序列化)"""
        return{
            "name":self.name,
            "age":self.age,
            "sex":self.sex,
            "scores":{subj:sc.to_dict() for subj,sc in self.scores.items()},
            "average_score":self.get_average_score(),
            "create_time":self.create_time
        }
    def save_to_file(self,filename=None):
        """保存到文件"""
        if not filename:
            filename = f"{self.name}_data.json"
        with open(filename,'w',encoding="utf-8") as f:
            json.dump(self.to_dict(),f,ensure_ascii=False,indent=2)
        return f"数据已保存到{filename}"
    @classmethod
    def from_dict(cls,data):
        """"从字典创建学生对象"""
        student = cls(data['name'],data['age'],data['sex'])
        for subject,score_data in data.get('scores',{}).items():
            student.scores[subject] = Score(subject,score_data['score'])
        return student

class ClassManager:
    """班级管理类 - 扩展：管理多个学生"""
    def __init__(self,class_name):
        self.class_name = class_name
        self.students = {}
    def add_student(self,student):
        """添加学生"""
        self.students[student.name] = student
        return f"学生{student.name}已经加入{self.class_name}"
    def remove_student(self,name):
        """移除学生"""
        if name in self.students:
            del self.students[name]
            return f"学生{name}已移除"
        return f"未找到学生{name}"
    def get_student(self,name):
        """查询学生"""
        return self.students.get(name,None)
    def get_class_average(self):
        """计算班级平均分"""
        if not self.students:
            return 0
        total_avg = sum(s.get_average_score() for s in self.students.values())
        # total_avg = 0
        # for s_name,s_object in self.students.items():
        #     total_avg += s_object.get_average_score()

        return round(total_avg/len(self.students),2)
    def get_ranking(self):
        """获取班级排名"""
        ranking = sorted(
            self.students.items(),
            key = lambda x:x[1].get_average_score(),
            reverse = True
        )
        result = f"\n{'='*50}\n{self.class_name}成绩排名\n{'='*50}"
        for rank,(name,student) in enumerate(ranking,1):
            result += f"\n第{rank}名：{name} - 平均分{student.get_average_score()}"
        result +=f"\n{'='*50}"
        return result
    def get_statistics(self):
        """班级统计信息"""
        if not self.students:
            return "班级暂无学生"
        averages = [s.get_average_score() for s in self.students.values()]
        stats = f"\n📊{self.class_name}统计信息"
        stats += f"\n学生总数：{len(self.students)}"
        stats += f"\n班级平均分：{self.get_class_average()}"
        stats += f"\n最高平均分:{max(averages)}"
        stats += f"\n最低平均分:{min(averages)}"
        return stats
    def save_class_data(self,filename=None):
        """保存班级数据"""
        if not filename:
            filename = f"{self.class_name}_data.json"
        data = {
            "class_name":self.class_name,
            "students":{name:stu.to_dict() for name,stu in self.students.items()}
        }
        with open(filename,'w',encoding="utf-8") as f:
            json.dump(data,f,ensure_ascii=False,indent=2)
        return f"班级数据已经保存到{filename}"

#=======================使用示例=======================

if __name__ == "__main__":
    print("="*60)
    print("🎓学生管理系统")
    print("="*60)

    #1.创建单个学生（向后兼容原功能）
    print("\n【1】创建学生对象")
    student1 = Student('姚闯',18,'男',{'数学':100})
    print(student1.introduce())
    print(student1.get_all_scores())

    #2.展示重构后的强大功能
    print("\n【2】多科目管理")
    student2 = Student('李华',17,'女',{
        '语文':85,
        '数学':92,
        '英语':78
    })
    print(student2.introduce())
    print(student2.get_all_scores())

    #3.更新成绩
    print("\n【3】更新成绩")
    print(student2.update_score('英语',88))
    print(student2.get_all_scores())

    #4.班级管理
    print("\n【4】班级管理功能")
    class_manager = ClassManager('高三（1）班')
    class_manager.add_student(student1)
    class_manager.add_student(student2)

    print(class_manager.get_ranking())
    print(class_manager.get_statistics())

    #5.数据持久化
    print("\n【5】数据保存")
    print(student1.save_to_file())
    print(class_manager.save_class_data())

    print("\n✅ 拆解-重构-扩展完成")
    print(" *拆解：Score类独立管理成绩")
    print(" *重构：Student类优化结构和验证")
    print(" *扩展：ClassManager管理整个班级")
