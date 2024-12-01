# Generated by Django 4.2.13 on 2024-12-01 12:43

from django.db import migrations, models
import tinymce.models


class Migration(migrations.Migration):

    dependencies = [
        ('quiz', '0008_remove_questiontruefalse_answer_answertruefalse'),
    ]

    operations = [
        migrations.AddField(
            model_name='exam',
            name='retry',
            field=models.SmallIntegerField(default=1, verbose_name='Số lượt thi tối đa'),
        ),
        migrations.AlterField(
            model_name='answer',
            name='answer_text',
            field=tinymce.models.HTMLField(blank=True, null=True, verbose_name='Câu trả lời'),
        ),
        migrations.AlterField(
            model_name='answertruefalse',
            name='clause',
            field=tinymce.models.HTMLField(blank=True, null=True, verbose_name='Mệnh đề'),
        ),
        migrations.AlterField(
            model_name='question',
            name='question_text',
            field=tinymce.models.HTMLField(blank=True, null=True, verbose_name='Câu hỏi'),
        ),
        migrations.AlterField(
            model_name='questionfill',
            name='answer',
            field=tinymce.models.HTMLField(blank=True, null=True, verbose_name='Đáp án đúng'),
        ),
        migrations.AlterField(
            model_name='questionfill',
            name='question_text',
            field=tinymce.models.HTMLField(blank=True, null=True, verbose_name='Câu hỏi'),
        ),
        migrations.AlterField(
            model_name='questiontruefalse',
            name='question_text',
            field=tinymce.models.HTMLField(blank=True, null=True, verbose_name='Câu hỏi'),
        ),
        migrations.AlterField(
            model_name='resultfill',
            name='answer',
            field=tinymce.models.HTMLField(blank=True, null=True, verbose_name='Câu trả lời'),
        ),
    ]
