from django.contrib import admin
from quiz.models.custom_user import * 
from quiz.models.quiz import * 
from django.contrib.auth.admin import UserAdmin 
import nested_admin 


@admin.register(CustomUser)
class CustomUserAdmin(UserAdmin):
    list_display = ('id', 'username', 'email', 'role', 'is_staff', 'is_superuser', 'is_active')
    search_fields = ('username', 'email', 'role')
    list_editable = ('role', 'is_staff', 'is_active', 'is_superuser')
    fieldsets = (
        (None, {'fields': ('username', 'email', 'password', 'role')}),
        ('Permissions', {'fields': ('is_staff', 'is_active', 'is_superuser', 'groups', 'user_permissions')}),
        ('Important dates', {'fields': ('last_login', 'date_joined')}),
    )


class AnswerInline(nested_admin.NestedTabularInline):
    model = Answer 
    extra = 4 


class QuestionInline(nested_admin.NestedTabularInline):
    model = Question 
    extra = 5
    inlines = [AnswerInline]


class AnswerTrueFalseInline(nested_admin.NestedTabularInline):
    model = AnswerTrueFalse
    extra = 4


class QuestionTrueFalseInline(nested_admin.NestedTabularInline):
    model = QuestionTrueFalse 
    extra = 5
    inlines = [AnswerTrueFalseInline]


class QuestionFillInline(nested_admin.NestedTabularInline):
    model = QuestionFill 
    extra = 5


@admin.register(Exam)
class ExamAdmin(nested_admin.NestedModelAdmin):
    inlines = [QuestionInline, QuestionTrueFalseInline, QuestionFillInline]
    list_display = ('title',)







class ResultDetailInline(admin.TabularInline):
    model = ResultDetail 
    extra = 1


@admin.register(Result)
class ResultAdmin(admin.ModelAdmin):
    inlines = [ResultDetailInline]
    search_fields = ['exam',]


@admin.register(Monitor)
class MonitorAdmin(admin.ModelAdmin):
    search_fields = ['id',]
